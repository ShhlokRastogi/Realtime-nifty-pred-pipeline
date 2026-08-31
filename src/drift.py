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
    
    conn = None
    try:
        # 1. Connect to DB and fetch resolved predictions
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 30000;")
        
        # Join forecasts with actual 5-candle realized average target
        cur.execute("""
            SELECT 
                f.source_datetime,
                f.current_realized_vol, 
                f.forecasted_vol_5h, 
                (
                    SELECT AVG(t.realized_vol_5)
                    FROM nifty_training_data t
                    WHERE t.datetime > f.source_datetime AND t.datetime <= f.target_datetime
                ) AS actual_future_vol
            FROM volatility_forecasts f
            WHERE f.target_datetime <= (SELECT MAX(datetime) FROM nifty_training_data)
            ORDER BY f.source_datetime DESC 
            LIMIT %s
        """, (window_hours,))
        
        rows = cur.fetchall()
        cur.close()
        
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
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate R2 Score (Percentage)
        r2 = r2_score(actuals, predictions) * 100.0
        
        # Calculate Directional Accuracy (1 = Rise, 0 = Fall)
        pred_rise = np.where(df_eval['predicted_future_vol'] > df_eval['current_vol'], 1, 0)
        actual_rise = np.where(df_eval['actual_future_vol'] > df_eval['current_vol'], 1, 0)
        current_accuracy = accuracy_score(actual_rise, pred_rise) * 100.0
        
        # Flag drift if accuracy drops below safety limit
        drift_detected = current_accuracy < critical_accuracy_threshold
        
        # 2. Log all metrics to database using the same connection (in a new transaction)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_drift_metrics 
            (evaluation_window_hours, mean_absolute_error, r2_score, directional_accuracy, accuracy_threshold, drift_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            int(window_hours), 
            float(mae), 
            float(r2), 
            float(current_accuracy), 
            float(critical_accuracy_threshold), 
            bool(drift_detected)
        ))
        conn.commit()
        cur.close()
        
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
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    monitor_accuracy_drift()