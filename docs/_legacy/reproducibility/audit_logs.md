# Audit Logs

**Audit logs** track your optimization decisions for reproducible research. The log records which data you used, which models you trained, and which experiments you decided to run—creating a complete, verifiable history of your project.

---

## Using Audit Logs in the Application

### The Workflow

**1. Train your model:**

- Add your experimental data

- Train a Gaussian Process model

- Check diagnostics (parity plot, Q-Q plot, etc.)

**2. Run acquisition:**

- Choose your acquisition strategy (Expected Improvement, UCB, etc.)

- Generate candidate experiments

- Review the suggestions

**3. Add to audit log:**

- **Desktop App:** When you're happy with a suggestion, click "Log to Audit Trail" in the notification window

- **Web App:** Click "Stage to Audit Log" in the Acquisition panel

- This marks it as a pending experiment you plan to run

- The log records: model used, acquisition settings, suggested point

**4. Add the experiment:**

- Run the experiment in your lab

- Use "Add Point" dialog to enter the results

- Or import from CSV

**5. Save your session:**

- The audit log is saved with your session

- Complete record of your optimization decisions

### Exporting the Audit Log

The audit log is stored in your session file. To review or share it:

**Desktop App:**

- File → Export Audit Log

- Saves as a Markdown file with a readable report

**Web App:**

- Click the export icon in the top toolbar

- Downloads audit log as Markdown

**What's included:**

- Timeline of all decisions

- Model configurations used

- Acquisition strategies and parameters

- Suggested experiments and results

- Notes you've added along the way

---

## Why Use Audit Logs?

**For reproducible research:**

- Prove you followed your methodology

- Show reviewers exactly what you did

- Document decisions weren't cherry-picked

- Support publications with verifiable records

**For your records:**

- Remember what worked and what didn't

- Track multiple optimization attempts

- Share methodology with colleagues

- Keep lab notebook records

**For collaboration:**

- Share decision rationale with team members

- Document protocol compliance

- Enable others to reproduce your work

---

## Advanced: Programmatic Access

For users writing Python scripts:

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()

# Add data and train model
session.add_experiment({'temp': 60, 'pressure': 5}, output=75.3)
session.train_model(backend='botorch')

# Stage acquisition to audit log
candidates = session.suggest_next(strategy='EI', n_candidates=5)
session.lock_acquisition(
    strategy='EI',
    candidates=candidates,
    notes="Batch 3 - targeting optimal region"
)

# Access audit log entries
for entry in session.audit_log.entries:
    print(f"{entry.timestamp}: {entry.entry_type}")
    
# Export audit log
session.audit_log.export('audit_log.json')
```

Audit logs are automatically saved with your session file.