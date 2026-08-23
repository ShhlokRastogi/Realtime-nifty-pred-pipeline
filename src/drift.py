import psycopg2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from config import DB_CONFIG

def monitor_accuracy_drift(window_hours=100, critical_accuracy_threshold=60.0):
    """
    Computes the MAE, R2 score, and Directional Accuracy of the model over the last N predictions.
    If Directional Accuracy falls below the safety threshold, triggers a concept drift alert.
    """
    print(f"Analyzing MAE, R2, and Directional Accuracy over the last {window_hours} hours...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # SQL: Join forecasts with actual realized training data 5 hours in the future
        cur.execute("""
            SELECT 
                f.datetime,
                f.current_realized_vol, 
                f.forecasted_vol_5h, 
                t.realized_vol_5 AS actual_future_vol
            FROM volatility_forecasts f
            INNER JOIN nifty_training_data t 
                ON t.datetime = f.datetime + interval '5 hours'
            ORDER BY f.datetime DESC 
            LIMIT %s
        """, (window_hours,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if len(rows) < 15:
            print("Insufficient matured forecast history in database to calculate metrics.")
            print("(Requires predictions that have already existed for at least 5 hours).")
            return
            
        # Parse data
        df_eval = pd.DataFrame(rows, columns=['datetime', 'current_vol', 'predicted_future_vol', 'actual_future_vol'])
        
        # Multiply actual future volatility by 100 to align with model percentage scale
        df_eval['actual_future_vol'] = df_eval['actual_future_vol'] * 100.0
        
        actuals = df_eval['actual_future_vol'].values
        predictions = df_eval['predicted_future_vol'].values
        
        # 1. Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(actuals, predictions)
        
        # 2. Calculate R2 Score (Percentage)
        r2 = r2_score(actuals, predictions) * 100.0
        
        # 3. Calculate Directional Accuracy (1 = Rise, 0 = Fall)
        pred_rise = np.where(df_eval['predicted_future_vol'] > df_eval['current_vol'], 1, 0)
        actual_rise = np.where(df_eval['actual_future_vol'] > df_eval['current_vol'], 1, 0)
        current_accuracy = accuracy_score(actual_rise, pred_rise) * 100.0
        
        # Flag drift if accuracy drops below safety limit
        drift_detected = current_accuracy < critical_accuracy_threshold
        
        # Log all metrics to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_drift_metrics 
            (evaluation_window_hours, mean_absolute_error, r2_score, directional_accuracy, accuracy_threshold, drift_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (window_hours, mae, r2, current_accuracy, critical_accuracy_threshold, drift_detected))
        conn.commit()
        cur.close()
        conn.close()
        
        print("\n" + "="*20 + " MODEL DRIFT & PERFORMANCE REPORT " + "="*20)
        print(f"Evaluation Window         : Last {len(df_eval)} resolved predictions")
        print(f"Mean Absolute Error (MAE) : {mae:.4f}% realized volatility")
        print(f"R-squared (R2) Score      : {r2:.2f}% (Variance explained)")
        print(f"Directional Accuracy      : {current_accuracy:.2f}%")
        print(f"Critical Safety Limit     : {critical_accuracy_threshold:.2f}%")
        print(f"Drift Status              : {'⚠️ DRIFT DETECTED - PERFORMANCE DEGRADATION' if drift_detected else '✅ STABLE (Performance is healthy)'}")
        print("="*60)
        
    except Exception as e:
        print(f"Error checking drift: {e}")

if __name__ == "__main__":
    monitor_accuracy_drift()