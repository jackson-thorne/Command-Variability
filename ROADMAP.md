Goal: Quantify Command

Process:
Take biomechanics data from Driveline internal databases
Filter all motion capture biomechanics to just use pitchers who were throwing a command bullpen trying to throw fastballs down the middle at 90% intensity or higher
Find the matching intended zones data for each pitch
Combine command and time sequenced biomechanics data to analyze factors influencing command
Looking to analyze mechanics through the lens of dynamical systems theory and chaos theory
Planning to analyze a Lyapunov's exponent for mechanics at each segment for every time in every throw
Look to quantify repeatability vs adaptability
This will ideally show how much repeatability or adaptability a pitcher needs, quantifying command will ideally outline where exactly command is being negatively impacted to make improvements, or what a pitcher does well.

This project takes command research from our Theia markerless biomechanics data and only uses pitchers who were throwing a command focused bullpen. Then the matching command results using our intended zones will show the result of each pitch. Here is the code to filter to only the command bullpens that we are looking for: 
SELECT trials.filename, trials.iz_trackman_pitch_id, trials.pulse_event_id, trials.handedness, sessions.height_meters, sessions.mass_kilograms, trials.session_trial
FROM trials
LEFT JOIN sessions ON sessions.session = trials.session
WHERE trials.tag IN ("iz", "IZ")
SELECT izd.*, pitches.id AS pitch_id FROM pitches
                    LEFT JOIN pitch_meta_data pmd ON pmd.id = pitches.pitch_meta_data_id
                    LEFT JOIN intended_zone_data izd ON izd.id = pitches.intended_zone_data_id
                    WHERE pitches.datetime_utc > '2024-12-01 00:00:00' AND pmd.pitch_type_id = 1
This should give you all of the session_trials that are needed in the biomech db, and all of the intended zones data. Can merge the IZ data onto the trials data and then only pull biomech data for those specific trials.
7:59
here's some python code that will pair and get the right time series biomech data
command_theia_df['iz_trackman_pitch_id'] = command_theia_df['iz_trackman_pitch_id'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
iz_poi_df = pd.merge(command_theia_df, iz_df, left_on='iz_trackman_pitch_id', right_on='pitch_id', how='left')
session_trial_list = ", ".join([f"'{str(x)}'" for x in iz_poi_df['session_trial'].unique()])

time_series_query = f"""SELECT * FROM joint_angles
LEFT JOIN joint_velos ON joint_velos.velo_id = joint_angles.angle_id
WHERE joint_angles.session_trial IN ({session_trial_list})"""

time_series_df = pd.read_sql(time_series_query, cnx)

Full Roadmap From O3
Below is an end-to-end technical roadmap you can hand directly to Cursor (or any modern code-gen IDE) to stand up the entire “Chaos-Theory Command” project from raw Driveline data to actionable results. 

 Every task is written so Cursor can emit runnable code/modules or shell commands with no additional context. 

 

0 Prerequisites & Environment 

Item 

Spec / Install command 

Python 

3.11 (Miniconda) 

Conda env 

conda create -n chaos_pitch python=3.11 numpy pandas polars pyarrow duckdb dask[complete] tsfresh nolds nolitsa scikit-learn xgboost matplotlib plotly hydra-core pytest black 

External libs 

pip install mlflow rich typer 

DB connector 

pip install mysql-connector-python (for HeidiSQL / MySQL) 

Create repo structure: 

chaos_command/ 
├── data/            # parquet dumps 
├── notebooks/       # exploratory work 
├── src/ 
│   ├── config/      # hydra YAML 
│   ├── utils/       # db.py, io.py, align.py 
│   ├── fe/          # feature_engineering.py 
│   ├── chaos/       # lyapunov.py, entropy.py 
│   ├── model/       # train.py, evaluate.py 
│   └── viz/         # plots.py 
└── tests/ 
  

 

1 Scope & Success Criteria 

Research question – quantify how repeatability (low early variance) vs adjustability (high late variance) affects miss-distance when pitchers aim MM fastballs. 

Produce: 

Metrics: SD profiles, funnel-in/out ratios, Lyapunov exponent, SampEn, correlation dimension. 

Predictive model of average miss distance. 

Visual report + CSV of player-level scores. 

Runtime target: single workstation, < 16 GB RAM per batch; total wall time < 2 h for ~30 command bullpens. 

(Problem statement & prior failed attempt in user doc ) 

 

2 Data Acquisition 

2.1 Identify Command Bullpen Sessions 

SQL pattern (Cursor converts to HeidiSQL syntax): 

WITH command_sessions AS ( 
  SELECT s.session_id, s.athlete_id, s.date 
  FROM sessions s 
  JOIN intended_zones iz ON iz.session_id = s.session_id 
  WHERE iz.target_zone = 'MM'              -- middle-middle 
    AND iz.pitch_type = 'FF'               -- fastball 
    AND iz.segment = 'command_bullpen'     -- specific tag used by R&D 
    AND iz.total_pitches = 15              -- exactly 15 attempts 
) 
SELECT * 
FROM command_sessions; 
  

Ask data analyst to confirm actual table/column names. 

2.2 Fetch Motion-Capture & Command Data 

SELECT m.*, iz.pitch_no, iz.miss_dist_inches 
FROM theia_capture m 
JOIN command_sessions cs USING (session_id) 
JOIN intended_zones iz 
  ON iz.session_id = cs.session_id 
  AND iz.pitch_no   = m.pitch_no; 
  

Export per session to compressed Parquet: 

# src/utils/db.py 
def fetch_to_parquet(query, out_path): ... 
  

Use chunked fetch (LIMIT … OFFSET …) to avoid OOM. 

 

3 Data Alignment & Cleaning 

Normalize timebase – resample markerless data to 250 Hz; align every throw on foot-plant (t = 0). 

Trim window – keep −300 ms → +300 ms around foot-plant (covers PKH → BR). 

Label key events – use peak events already in Theia (pkh, FP, MER, BR); fallback to heuristics if missing. 

Deduplicate & QC – drop throws with > 10% missing markers; assert miss_dist present. 

src/align.py implements these. 

 

4 Feature Engineering 

Stage 

Features (per frame unless noted) 

Raw kinematics 

joint angles, angular velocities (pelvis, trunk, shoulder, elbow, wrist) 

Phase SDs 

SD at PKH, FP, MER, BR (per joint / velocity) 

Funnel ratios 

SD₁/SD₂ where SD₂ is chronologically later & up the chain (Lambert metric) 

Chaos 

Largest Lyapunov exponent (LLE), SampEn, correlation dimension on 3-D state vector [pelvis_rot, trunk_rot, elbow_flex] 

Temporal embeddings 

time-delay embedding (m=10, τ=4 ms) for LLE 

src/fe/feature_engineering.py produces a Parquet Δ. 

 

5 Chaos & Dynamical Systems Analysis 

from nolitsa import lyapunov 
LLE = lyapunov.max_lyapunov(ts, emb_dim=10, matrix_dim=20) 
  

Repeat for each throw, then aggregate: 

Repeatability index = mean SD at FP + mean SD upper-body @ MER 

Adjustability index = mean SD ratio (late/early) for scap_retraction & pronation 

Chaos score = median LLE across all joints 

Produce athlete/session summary table. 

 

6 Modeling 

Train/Test split – leave-one-athlete-out CV (27 athletes → 27 folds). 

Baseline – LinearRegression on repeatability & adjustability indices. 

ML – XGBRegressor on full feature grid. 

Metric – RMSE on miss_dist_inches; target < 1.5 inches RMSE. 

Log to MLflow. 

 

7 Visualization & Reporting 

Funnel plots – variance trajectory PKH→BR per athlete. 

Chaos vs Miss scatter – highlight funnel-in vs funnel-out clusters. 

Interactive HTML – Plotly dash or Jupyter Book. 

src/viz/plots.py. 

 

8 Automation Pipeline 

Create src/main.py with Typer CLI: 

chaoscli pull-data        # Section 2 
chaoscli align            # Section 3 
chaoscli fe               # Section 4 
chaoscli chaos-analysis   # Section 5 
chaoscli train-model      # Section 6 
chaoscli report           # Section 7 
  

Hydra config (config/config.yaml) holds DB creds & paths. 

Add Makefile: 

setup:  ## create env 
    conda env create -f environment.yml 
run-all: 
    chaoscli pull-data && chaoscli align && chaoscli fe && chaoscli chaos-analysis && chaoscli train-model && chaoscli report 
  

Set up pre-commit: black, pylint, pytest. 

 

9 Timeline (sprint outline) 

Week 

Deliverable 

1 

Confirm schema, write & test SQL extract, Parquet dump on 1 session 

2 

Alignment & QC utils, run on all sessions 

3 

Feature engineering + chaos metrics validated on sample 

4 

Full data run, athlete summary table 

5 

Modeling & MLflow tracking 

6 

Visualization drafts, internal tech review 

7 

Final report, README, code hand-off 

 

10 Risk & Mitigation 

Risk 

Mitigation 

Data volume 

Session-wise extracts, Parquet + Polars/Dask; no full CSV in memory 

Missing markers 

Interpolate ≤ 5 frames; else drop throw 

Small N (≈30) 

Focus on within-athlete metrics; LOAO CV 

Cursor hallucinations 

Unit tests with synthetic 3-throw fixture; docstrings w/ I/O examples 

 

11 Hand-off Package 

environment.yml 

README with CLI examples & expected outputs 

Folder sample_output/ – one athlete JSON & plots 

Markdown summary of findings for coaches 

 

Final note 

This roadmap mirrors the funnel-in/funnel-out framework described by Lambert & Pelletier and explicitly replaces your earlier “load-18 M rows & hope” approach with chunked Parquet + staged processing, solving the freeze you hit at “STARTING COMPLETE CHAOS THEORY ANALYSIS” . 

Hand this file to Cursor; every section is directly translatable into code, configs, or shell commands. Good luck turning chaos into command! 
