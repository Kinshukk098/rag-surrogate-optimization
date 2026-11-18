
import numpy as np

from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
)


class RBFNetworkRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_centers=20,
        alpha=1.0,
        random_state=42
    ):
        self.n_centers = n_centers
        self.alpha = alpha
        self.random_state = random_state

    def _rbf_features(self, X):
        distances = cdist(X, self.centers_, metric="euclidean")
        return np.exp(
            -(distances ** 2) / (2 * self.sigma_ ** 2)
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array.")

        # Do not request more centres than available unique samples.
        unique_X = np.unique(X, axis=0)
        effective_centers = min(
            int(self.n_centers),
            len(unique_X),
            len(X)
        )

        if effective_centers < 1:
            raise ValueError("At least one RBF centre is required.")

        self.n_centers_ = effective_centers

        self.kmeans_ = KMeans(
            n_clusters=effective_centers,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(X)

        self.centers_ = self.kmeans_.cluster_centers_

        center_distances = cdist(
            self.centers_,
            self.centers_,
            metric="euclidean"
        )

        nonzero_distances = center_distances[
            center_distances > 1e-12
        ]

        if len(nonzero_distances) == 0:
            # All input points are identical; use a safe positive width.
            self.sigma_ = 1.0
        else:
            self.sigma_ = float(np.median(nonzero_distances))

        Phi = self._rbf_features(X)

        self.output_model_ = Ridge(alpha=self.alpha)
        self.output_model_.fit(Phi, y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Phi = self._rbf_features(X)
        return self.output_model_.predict(Phi)


def create_models(random_state=42, model_config=None):

    model_config = model_config or {}

    linear_cfg = model_config.get("linear_regression", {})
    poly_cfg = model_config.get("polynomial_regression", {})
    rf_cfg = model_config.get("random_forest", {})
    rbf_cfg = model_config.get("rbf_network", {})
    gp_cfg = model_config.get("gaussian_process", {})

    models = {
        "linear_regression": LinearRegression(
            **linear_cfg
        ),

        "polynomial_regression": Pipeline([
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=poly_cfg.get("degree", 2),
                    include_bias=False
                )
            ),
            (
                "regressor",
                Ridge(alpha=poly_cfg.get("alpha", 1.0))
            )
        ]),

        "random_forest": RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 300),
            max_features=rf_cfg.get("max_features", 1.0),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=random_state,
            n_jobs=-1
        ),

        "rbf_network": RBFNetworkRegressor(
            n_centers=rbf_cfg.get("n_centers", 20),
            alpha=rbf_cfg.get("alpha", 1.0),
            random_state=random_state
        ),

        "gaussian_process": GaussianProcessRegressor(
            kernel=(
                ConstantKernel(
                    1.0,
                    constant_value_bounds=(1e-2, 1e2)
                )
                * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e2)
                )
                + WhiteKernel(
                    noise_level=0.01,
                    noise_level_bounds=(1e-5, 1e1)
                )
            ),
            normalize_y=True,
            n_restarts_optimizer=gp_cfg.get(
                "n_restarts_optimizer", 2
            ),
            random_state=random_state
        )
    }

    return models
