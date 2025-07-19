**ABSTRACT**
	
This project builds a reproducible ML Ops workflow that tests, trains and serves inference for a logistic regression digit classifier.

Using GitHub Actions, every push triggers (1) PyTest verification, (2) model training with hyper parameters read from JSON, and (3) inference that re uses the model artifact produced in the previous job.

Branch isolation—main → classification_branch → test_branch → inference_branch—keeps each phase self contained while preserving lineage.

The result is a fully automated CI/CD pipeline that prints accuracy, uploads model_train.pkl, and guarantees that code, data, and artifacts stay in sync across environments.

# ML Ops Assignment 2
INTRODUCTION
Modern machine learning projects must be repeatable, testable, and continuously deployable. Assignment 2 translates those principles into practice by asking us to:
•	Parameterise training — hyper parameters (C, solver, max_iter, test_size, random_state) are stored in config/config.json, eliminating hidden state.
•	Codify quality gates — unit & integration tests in tests/ assert that the config loads, the LogisticRegression model fits, and accuracy crosses an empirical threshold.
•	Automate the lifecycle — three GitHub Actions workflows turn every commit into a miniature release cycle:
1.	test.yml → runs PyTest.
2.	train.yml → fits the model and uploads model_train.pkl.
3.	inference.yml → downloads that artifact and emits predictions.
•	Preserve history with linear branching — each phase lives on its own branch without merging back to main, exactly as mandated on page 3 of the brief. 
Together these elements deliver the learning outcomes listed on pages 1 2—“parameterise training,” “write tests,” and “design CI/CD with GitHub Actions.” 
