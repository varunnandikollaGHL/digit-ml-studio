# BITS Virtual Lab - actual execution checklist

This project was generated and evaluated in a separate environment. **A GitHub Actions run does not count as BITS Virtual Lab evidence.**

1. Open the official BITS Virtual Lab assigned to you, and download or clone this repository:
   `git clone https://github.com/varunnandikollaGHL/digit-ml-studio.git`
2. Change into the project folder: `cd digit-ml-studio`.
3. Check Python: `python --version` (Python 3.12 preferred).
4. Install the project's dependencies: `python -m pip install -r requirements.txt`.
5. Run `python train_models.py`. The script prints scores for all five classifiers and writes `model/*.joblib`, `test_data.csv`, `metrics.csv`, and `metadata.json`.
6. Run `python -m unittest discover -s tests -v` and confirm the checks pass.
7. Run `streamlit run app.py` if the lab allows local web servers.
8. Capture **one real screenshot** that visibly shows the BITS Virtual Lab environment and the completed training/output. Do not substitute a GitHub Actions screenshot.
9. Insert that screenshot in the reserved area on page 2 of your submission document.

The source dataset loads without external downloads. The required submission is one PDF containing: (1) clickable GitHub URL, (2) clickable deployed Streamlit URL, (3) genuine Virtual Lab screenshot, and (4) full README content.

If the BITS lab cannot be accessed, the assignment tells students to email csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in with subject `NSP1 ML Assignment 2: BITS Lab issue`.
