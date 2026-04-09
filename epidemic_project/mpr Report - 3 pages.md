# MPR Report (Condensed): GMM for Hospital Operations and Patient Flows

## 1. Purpose and Context
This report presents a concise, max-3-page explanation of how Gaussian Mixture Models (GMM) are used in this project to improve epidemic-time hospital operations.

The project goal is not only to simulate epidemic spread, but to make hospital decisions earlier and more intelligently. Specifically, GMM is used to:
1. Detect multiple patient arrival patterns.
2. Identify latent hospital stress modes.
3. Convert those patterns into pre-allocation and response actions.

This transforms the workflow from retrospective analysis into anticipatory operations planning.

## 2. GMM Concept in Simple Terms
A Gaussian Mixture Model assumes observed data comes from several overlapping groups, where each group follows a Gaussian (normal) distribution.

The model is:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

Where:
1. $K$ is number of latent groups (clusters).
2. $\pi_k$ is weight of group $k$.
3. $\mu_k$ and $\Sigma_k$ are mean and covariance of group $k$.

Why this matters:
1. Hospital systems rarely have hard boundaries between states.
2. Stress and arrivals overlap in time and intensity.
3. GMM handles uncertainty better than rigid clustering methods.

## 3. Why GMM Fits This Project
Epidemic hospital data is multi-modal and non-linear:
1. Arrivals may peak in different windows (morning, afternoon, late).
2. Similar utilization can correspond to different operational risk.
3. Overload emerges gradually, not as a single threshold jump.

GMM is effective here because it:
1. Finds multiple latent behavior regimes.
2. Supports probabilistic interpretation of states.
3. Produces interpretable cluster summaries for planning.
4. Works with simulation-scale tabular data.

## 4. Data Signals Used
The pipeline uses healthcare simulation outputs, including:
1. Day index.
2. New infections.
3. Hospitalized count.
4. Occupancy.
5. Patients turned away.
6. Healthcare utilization.
7. Overload indicator.

If event-level admission timestamps are missing, the workflow estimates admissions from daily dynamics and creates an hourly arrival profile.

## 5. Pipeline Application of GMM

### 5.1 Arrival Behavior Clustering
Input features:
1. Cyclical hour features ($hour\_sin$, $hour\_cos$).
2. Hourly arrivals.
3. Rolling 6-hour arrival average.
4. Healthcare utilization context.

Result:
1. BIC-based selection of cluster count.
2. Arrival behavior labels such as low flow, moderate flow, high flow, and surge windows.

Operational value:
1. Better shift planning.
2. Better ED/ward staffing alignment.
3. Early surge preparation.

### 5.2 Hospital Stress Mode Clustering
Input features:
1. Utilization.
2. Occupancy.
3. Hospitalized count.
4. Turned-away patients.
5. New infections.
6. Estimated admissions.
7. Numeric overload flag.

Result:
1. BIC-based stress mode identification.
2. Ordered operational modes from stable to critical.

Operational value:
1. State-aware planning instead of single-metric monitoring.
2. Earlier warning before visible collapse.

### 5.3 Predictive Pre-Allocation
From cluster outputs, the workflow:
1. Estimates likely next stress mode.
2. Computes expected admissions.
3. Pre-allocates beds, nurses, ICU slots, and PPE kits.

Operational value:
1. Reduces reactive firefighting.
2. Improves resource timing and utilization.

### 5.4 Step 7 Decision Engine
The Step 7 extension maps predictions into action:
1. Alert levels: Normal, Watch, Warning, Critical.
2. Priority action bundles for staffing, triage, and communication.
3. Exportable action-plan artifact for downstream agents.

Operational value:
1. Direct link from analytics to execution.
2. Better coordination across hospital and policy actors.

## 6. Significance to the Project
This GMM layer is strategically important because it upgrades the project in five ways:

1. From descriptive to predictive:
   - The system does not only show current burden; it anticipates near-term stress.

2. From single-metric to multi-signal intelligence:
   - Joint modeling of admissions, utilization, occupancy, and turn-away improves realism.

3. From static thresholds to latent state detection:
   - Operational modes capture nuanced transitions that fixed thresholds may miss.

4. From model output to practical action:
   - Step 7 provides executable plans, not just charts and cluster IDs.

5. From isolated analysis to multi-agent integration:
   - Structured outputs can be consumed by hospital, logistics, and communication agents.

## 7. Outputs and Their Practical Use
The notebook produces four key files:
1. patient_arrival_behaviors_gmm.csv
2. hospital_stress_modes_gmm.csv
3. resource_preallocation_recommendations.csv
4. hospital_operations_action_plan.csv

These outputs support:
1. Monitoring dashboards.
2. Shift and bed planning.
3. Emergency escalation protocols.
4. Inter-hospital coordination logic.

## 8. Assumptions and Limits
1. Synthetic admission timing is an approximation when true timestamps are absent.
2. Mode transition logic is simple and may not capture long temporal memory.
3. Mode labels are operational abstractions, not clinical labels.
4. Cluster quality depends on feature design and data quality.

## 9. Recommended Next Improvements
1. Ingest real event-level admission timestamps.
2. Calibrate alert thresholds with domain experts and historical events.
3. Add temporal probabilistic models for transition dynamics.
4. Report uncertainty bands for allocation recommendations.
5. Validate impact using KPIs such as wait time, turn-away rate, and surge response delay.

## 10. Conclusion
GMM is central to this project’s hospital intelligence layer. It identifies hidden arrival and stress patterns, supports near-term forecasting, and enables practical pre-allocation and escalation actions. In short, it strengthens readiness, resource efficiency, and decision quality under epidemic uncertainty while preserving compatibility with a multi-agent operational framework.
