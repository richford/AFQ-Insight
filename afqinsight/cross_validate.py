"""Include functions to validate model performance using cross-validation."""

import copy
import deepdish.io as ddio
import hashlib
import json
import pandas as pd
import numpy as np
import os
import pickle

from joblib import delayed, Parallel
from sklearn.base import clone, is_classifier
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts, _fit_and_score
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import indexable

__all__ = ["cross_validate_checkpoint"]


def _serialize_estimator_params(estimator_params):
    # The estimator might be a pipeline, in which case, we want to pop the
    # objects out and leave only the params
    params = copy.deepcopy(estimator_params)
    if "steps" in params:
        steps = params.pop("steps")
        step_names, _ = tuple(zip(*steps))
        for s in step_names:
            params.pop(s)

        params["steps"] = step_names

    return params


def _fit_and_score_ckpt(
    workdir=None, checkpoint=True, force_refresh=False, **fit_and_score_kwargs
):
    """Fit estimator and compute scores for a given dataset split.

    This function wraps
    :func:`sklearn:sklearn.model_selection._validation._fit_and_score`,
    while also saving checkpoint files containing the estimator, paramters,
    This is useful if fitting and scoring is costly or if it is being
    performed within a large cross-validation experiment.

    In avoid collisions with scores computed for other CV splits, this
    function computes a hash from a nested dictionary containing all keyword
    arguments as well as estimator parameters. It then saves the scores and
    parameters in <hash>_params.h5 and the estimator itself in
    <hash>_estimator.pkl

    Parameters
    ----------
    workdir : path-like object, default=None
        A string or :term:`python:path-like-object` indicating the directory
        in which to store checkpoint files

    checkpoint : bool, default=True
        If True, checkpoint the parameters, estimators, and scores.

    force_refresh : bool, default=False
        If True, recompute scores even if the checkpoint file already exists.
        Otherwise, load scores from checkpoint files and return.

    **fit_and_score_kwargs : kwargs
        Key-word arguments passed to
        :func:`sklearn:sklearn.model_selection._validation._fit_and_score`

    Returns
    -------
    train_scores : dict of scorer name -> float
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None
        The parameters that have been evaluated.

    estimator : estimator object
        The fitted estimator
    """
    if not checkpoint:
        return _fit_and_score(**fit_and_score_kwargs)

    if workdir is None:
        raise ValueError(
            "If checkpoint is True, you must supply a working directory "
            "through the ``workdir`` argument."
        )

    estimator = fit_and_score_kwargs.pop("estimator", None)
    estimator_params = _serialize_estimator_params(estimator.get_params())
    all_params = {
        "estimator_params": estimator_params,
        "fit_and_score_kwargs": fit_and_score_kwargs,
    }

    cv_hash = hashlib.md5(
        json.dumps(all_params, sort_keys=True, ensure_ascii=True, default=str).encode()
    ).hexdigest()

    h5_file = os.path.join(workdir, cv_hash + "_params.h5")
    pkl_file = os.path.join(workdir, cv_hash + "_estimator.pkl")

    if not force_refresh and os.path.exists(h5_file):
        ckpt_dict = ddio.load(h5_file)

        scores = ckpt_dict["scores"]

        if fit_and_score_kwargs.get("return_estimator", False):
            with open(pkl_file, "rb") as fp:
                estimator = pickle.load(fp)

            scores.append(estimator)

        return scores
    else:
        scores = _fit_and_score(estimator, **fit_and_score_kwargs)
        os.makedirs(workdir, exist_ok=True)
        if fit_and_score_kwargs.get("return_estimator", False):
            estimator = scores[-1]
            with open(pkl_file, "wb") as fp:
                pickle.dump(estimator, fp)

            ckpt_scores = scores[:-1]
            if isinstance(estimator, Pipeline):
                model = estimator.steps[-1]
            else:
                model = estimator

            estimator_params = _serialize_estimator_params(estimator.get_params())
            fitted_params = {
                "alpha_": getattr(model, "alpha_", None),
                "alphas_": getattr(model, "alphas_", None),
                "l1_ratio_": getattr(model, "l1_ratio_", None),
                "mse_path_": getattr(model, "mse_path_", None),
                "scoring_path_": getattr(model, "scoring_path_", None),
                "intercept_": getattr(model, "intercept_", None),
                "coef_": getattr(model, "coef_", None),
            }
        else:
            estimator_params = None
            fitted_params = None
            ckpt_scores = scores

        fit_and_score_kwargs.pop("X")
        fit_and_score_kwargs.pop("y")

        if "scorer" in fit_and_score_kwargs:
            fit_and_score_kwargs["scorer"] = list(fit_and_score_kwargs["scorer"].keys())

        ckpt_dict = {
            "scores": ckpt_scores,
            "fit_and_score_kwargs": fit_and_score_kwargs,
            "estimator_params": estimator_params,
            "fitted_params": fitted_params,
        }

        ddio.save(h5_file, ckpt_dict)
        return scores


def cross_validate_checkpoint(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
    workdir=None,
    checkpoint=True,
    force_refresh=False,
    serialize_cv=False,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    This is a copy of :func:`sklearn:sklearn.model_selection.cross_validate`
    that uses :func:`_fit_and_score_ckpt` to checkpoint scores and estimators
    for each CV split.
    Read more in the :ref:`sklearn user guide <sklearn:multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`sklearn:GroupKFold`).

    scoring : str, callable, list/tuple, or dict, default=None
        A single str (see :ref:`sklearn:scoring_parameter`) or a callable
        (see :ref:`sklearn:scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`sklearn:multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - an sklearn `CV splitter <https://scikit-learn.org/stable/glossary.html#term-cv-splitter>`_,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.
        Refer :ref:`sklearn user guide <sklearn:cross_validation>` for the
        various cross-validation strategies that can be used here.

    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib:joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`sklearn Glossary <sklearn:n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

    workdir : path-like object, default=None
        A string or :term:`python:path-like-object` indicating the directory
        in which to store checkpoint files

    checkpoint : bool, default=True
        If True, checkpoint the parameters, estimators, and scores.

    force_refresh : bool, default=False
        If True, recompute scores even if the checkpoint file already exists.
        Otherwise, load scores from checkpoint files and return.

    serialize_cv : bool, default=False
        If True, do not use joblib.Parallel to evaluate each CV split.

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.

    Examples
    --------
    >>> import shutil
    >>> import tempfile
    >>> from sklearn import datasets, linear_model
    >>> from afqinsight import cross_validate_checkpoint
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate_checkpoint(lasso, X, y, cv=3, checkpoint=False)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])

    Multiple metric evaluation using ``cross_validate``, an estimator
    pipeline, and checkpointing (please refer the ``scoring`` parameter doc
    for more information)

    >>> tempdir = tempfile.mkdtemp()
    >>> scaler = StandardScaler()
    >>> pipeline = make_pipeline(scaler, lasso)
    >>> scores = cross_validate_checkpoint(pipeline, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True, checkpoint=True,
    ...                         workdir=tempdir, return_estimator=True)
    >>> shutil.rmtree(tempdir)
    >>> print(scores['test_neg_mean_squared_error'])
    [-2479.2... -3281.2... -3466.7...]
    >>> print(scores['train_r2'])
    [0.507... 0.602... 0.478...]

    See Also
    --------
    sklearn.model_selection.cross_val_score:
        Run cross-validation for single metric evaluation.
    sklearn.model_selection.cross_val_predict:
        Get predictions from each split of cross-validation for diagnostic
        purposes.
    sklearn.metrics.make_scorer:
        Make a scorer from a performance metric or loss function.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    if serialize_cv:
        scores = [
            _fit_and_score_ckpt(
                workdir=workdir,
                checkpoint=checkpoint,
                force_refresh=force_refresh,
                estimator=clone(estimator),
                X=X,
                y=y,
                scorer=scorers,
                train=train,
                test=test,
                verbose=verbose,
                parameters=None,
                fit_params=fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in cv.split(X, y, groups)
        ]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
        scores = parallel(
            delayed(_fit_and_score_ckpt)(
                workdir=workdir,
                checkpoint=checkpoint,
                force_refresh=force_refresh,
                estimator=clone(estimator),
                X=X,
                y=y,
                scorer=scorers,
                train=train,
                test=test,
                verbose=verbose,
                parameters=None,
                fit_params=fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in cv.split(X, y, groups)
        )

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    ret = {}
    ret["fit_time"] = np.array(fit_times)
    ret["score_time"] = np.array(score_times)

    if return_estimator:
        ret["estimator"] = fitted_estimators

    for name in scorers:
        ret["test_%s" % name] = np.array(test_scores[name])
        if return_train_score:
            key = "train_%s" % name
            ret[key] = np.array(train_scores[name])

    return ret


def stratified_crossvalidate(
    covariate_df,
    pipeline_estimator=None,
    cv_splitter=StratifiedShuffleSplit(n_splits=1, test_size=0.25),
):
    """
    Given multiple covariates in covariate_df, returns training and test splits
    such that covariate distributions are similar across both training and test splits

    Parameters
    ----------
    covariate_df : pandas dataframe of shape (n_samples, n_features)

    pipeline_estimator : default=None
        Custom pipeline not yet supported

    cv_splitter : A cv object default=StratifiedShuffleSplit
        A scikit-learn object for splitting the dataset into
        train/test set such that splits are stratified

    Yields
    ------
    train : ndarray
        The training set indices for one shuffle split
    test : ndarray
        The testing set indices for one shuffle split
    covariate: ndarray
        The composite covariate used to stratify training and test splits

    """

    # Check covariate_df is a dataframe
    if not isinstance(covariate_df, pd.DataFrame):
        raise TypeError

    X = covariate_df.to_numpy()

    if pipeline_estimator is None:
        pipeline_estimator = make_pipeline(
            KNeighborsTransformer(n_neighbors=10, mode="distance"),
            Isomap(n_components=5, neighbors_algorithm="auto"),
        )

    X_reduced = pipeline_estimator.fit_transform(X)
    cluster_stratify = KMeans(n_clusters=5).fit_predict(X_reduced)

    for foldno, (train_index, test_index) in enumerate(
        cv_splitter.split(X, cluster_stratify)
    ):
        print("Fold No: {}".format(foldno))
        train_clusters, train_freq = np.unique(
            cluster_stratify[train_index], return_counts=True
        )
        test_clusters, test_freq = np.unique(
            cluster_stratify[test_index], return_counts=True
        )
        print("Training data: Cluster frequency")
        print(train_clusters)
        print(train_freq)
        print(test_clusters)
        print(test_freq)

    train_keys = covariate_df.index[train_index]
    test_keys = covariate_df.index[test_index]

    return train_keys, test_keys, cluster_stratify
