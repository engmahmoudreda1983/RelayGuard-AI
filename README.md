# RelayGuard AI

Protection relay condition and risk monitor. Part of the PowerGuard AI platform. Confidential and proprietary.

**Streamlit main file:** `RelayGuard_AI.app.py`
**Secrets:** `APP_USERNAME`, `APP_PASSWORD`

## Pages
- **Substation Overview:** results collected from the relay page, or the demo substation (15 relays).
- **Relay Risk & FMEA:** latest evidence of one relay → risk for 7 failure modes, overall risk, FMEA and RPN.
- **Disturbance Records (AI):** upload COMTRADE files (.cfg + .dat); waveforms, extracted features and the AI anomaly score.
- **OMICRON Test Review:** test points against tolerance, drift between tests, downloadable HTML test report.
- **Alarms & Events:** event log (SER) counted in the engine's windows, event timeline.
- **Breaker Timing:** breaker opening time against rated, trend.
- **Combined Report** and **Historical Study** (demo or client CSV files).

## AI
`record_anomaly_model.pkl`: Isolation Forest trained on 694 real disturbance records (fault current, pre-fault noise index).
Retrain with `python train_record_model.py <features.xlsx>`; the training data is not stored in the repository.

## Files
- `comtrade_features.py`: COMTRADE reader and record features.
- `relayguard_core.py`: engine (test-point accuracy and drift, self-supervision, communication, secondary circuits, breaker timing, age, test interval; FMEA; daily historical replay; demo substation).
