"""Kalman filter used by ByteTrack.

A constant-velocity filter over the ``[x, y, a, h]`` bounding-box state with
optional aspect-ratio adaptation, score weighting, and gap recovery.
"""

import numpy as np
import scipy.linalg
from .modules.score_noise import observation_noise_scale
from .modules.aspect_noise import adapt_aspect_scale
from .modules.gap_state import reinit_after_gap


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """ByteTrack filter with optional aspect-ratio process-noise adaptation."""

    def __init__(self, enable_score_weight=False, enable_gap_recovery=False,
                 gap_covariance_threshold=10, gap_reset_threshold=30,
                 min_box_height=1e-3,
                 enable_aspect_adaptive=False, aspect_scale_low=1.0,
                 aspect_scale_high=10.0, aspect_gate_thresh=2.5,
                 aspect_ema_alpha=0.5):
        if not (0 <= gap_covariance_threshold < gap_reset_threshold):
            raise ValueError(
                "gap thresholds must satisfy 0 <= covariance < reset")
        if not np.isfinite(min_box_height) or min_box_height <= 0:
            raise ValueError("min_box_height must be positive")
        if aspect_scale_low < 1.0:
            raise ValueError("aspect_scale_low must be >= 1.0")
        if aspect_scale_high < aspect_scale_low:
            raise ValueError("aspect_scale_high must be >= aspect_scale_low")
        if not aspect_gate_thresh > 0:
            raise ValueError("aspect_gate_thresh must be positive")
        if not 0.0 < aspect_ema_alpha <= 1.0:
            raise ValueError("aspect_ema_alpha must be in (0, 1]")

        ndim, dt = 4, 1.0
        self._dt = dt
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for index in range(ndim):
            self._motion_mat[index, ndim + index] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._min_box_height = float(min_box_height)

        self._enable_score_weight = bool(enable_score_weight)
        self._enable_gap_recovery = bool(enable_gap_recovery)
        self._gap_covariance_threshold = int(gap_covariance_threshold)
        self._gap_reset_threshold = int(gap_reset_threshold)

        self._enable_aspect_adaptive = bool(enable_aspect_adaptive)
        self._aspect_scale_low = float(aspect_scale_low)
        self._aspect_scale_high = float(aspect_scale_high)
        self._aspect_gate_thresh = float(aspect_gate_thresh)
        self._aspect_ema_alpha = float(aspect_ema_alpha)

    def _safe_height(self, height):
        return max(abs(float(height)), self._min_box_height)

    def initiate(self, measurement):
        """Create an eight-dimensional state from ``[x, y, a, h]``."""
        measurement = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[measurement, np.zeros_like(measurement)]
        height = self._safe_height(measurement[3])
        std = [
            2 * self._std_weight_position * height,
            2 * self._std_weight_position * height,
            1e-2,
            2 * self._std_weight_position * height,
            10 * self._std_weight_velocity * height,
            10 * self._std_weight_velocity * height,
            1e-5,
            10 * self._std_weight_velocity * height,
        ]
        return mean, np.diag(np.square(std))

    def _motion_covariance(self, mean, aspect_scale=1.0):
        """Build Q, scaling only the aspect-ratio state and velocity."""
        height = self._safe_height(mean[3])
        std_pos = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-2 * aspect_scale,
            self._std_weight_position * height,
        ]
        std_vel = [
            self._std_weight_velocity * height,
            self._std_weight_velocity * height,
            1e-5 * aspect_scale,
            self._std_weight_velocity * height,
        ]
        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def predict(self, mean, covariance, aspect_scale=None):
        """Predict position and size with the configured transition model."""
        if aspect_scale is None:
            aspect_scale = 1.0
        motion_covariance = self._motion_covariance(mean, aspect_scale)
        predicted_mean = np.dot(self._motion_mat, mean)
        predicted_covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T))
        predicted_covariance += motion_covariance
        return predicted_mean, predicted_covariance

    def multi_predict(self, mean, covariance, aspect_scales=None):
        """Run prediction independently for each track."""
        count = len(mean)
        if aspect_scales is None:
            aspect_scales = np.ones(count)
        for index in range(count):
            mean[index], covariance[index] = self.predict(
                mean[index], covariance[index],
                aspect_scale=aspect_scales[index])
        return mean, covariance

    def project(self, mean, covariance, measurement_noise_scale=1.0):
        """Project state and covariance into measurement space."""
        if (not np.isfinite(measurement_noise_scale) or
                measurement_noise_scale <= 0):
            raise ValueError("measurement_noise_scale must be positive")
        height = self._safe_height(mean[3])
        std = [
            self._std_weight_position * height,
            self._std_weight_position * height,
            1e-1,
            self._std_weight_position * height,
        ]
        innovation_covariance = (
            np.diag(np.square(std)) * measurement_noise_scale)
        projected_mean = np.dot(self._update_mat, mean)
        projected_covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return projected_mean, projected_covariance + innovation_covariance

    def _adapt_aspect_scale(self, normalized_aspect, aspect_scale):
        if not self._enable_aspect_adaptive:
            return aspect_scale
        return adapt_aspect_scale(normalized_aspect, aspect_scale, self._aspect_scale_low,
                                  self._aspect_scale_high, self._aspect_gate_thresh, self._aspect_ema_alpha)

    def _update_impl(self, mean, covariance, measurement,
                     measurement_noise_scale, aspect_scale=1.0):
        projected_mean, projected_covariance = self.project(
            mean, covariance,
            measurement_noise_scale=measurement_noise_scale)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_covariance, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))
        new_aspect_scale = aspect_scale
        if self._enable_aspect_adaptive:
            normalized = scipy.linalg.solve_triangular(
                chol_factor, innovation, lower=True, check_finite=False)
            new_aspect_scale = self._adapt_aspect_scale(
                float(normalized[2]), aspect_scale)
        return new_mean, new_covariance, new_aspect_scale

    def update(self, mean, covariance, measurement, score=None,
               aspect_scale=None):
        """Correct a predicted state with one detection."""
        if aspect_scale is None:
            aspect_scale = self._aspect_scale_low
        return self._update_impl(
            mean, covariance, measurement,
            measurement_noise_scale=1.0, aspect_scale=aspect_scale)

    def update_with_score(self, mean, covariance, measurement, score,
                          aspect_scale=None):
        """Correct the state while modestly increasing R for low scores."""
        if aspect_scale is None:
            aspect_scale = self._aspect_scale_low
        score = float(np.clip(score, 0.0, 1.0))
        noise_scale = observation_noise_scale(score)
        return self._update_impl(
            mean, covariance, measurement,
            measurement_noise_scale=noise_scale, aspect_scale=aspect_scale)

    def _update_for_score(self, mean, covariance, measurement, score,
                          aspect_scale=None):
        if aspect_scale is None:
            aspect_scale = self._aspect_scale_low
        if self._enable_score_weight and score is not None:
            return self.update_with_score(
                mean, covariance, measurement, score=score,
                aspect_scale=aspect_scale)
        return self.update(
            mean, covariance, measurement, score=score,
            aspect_scale=aspect_scale)

    def reinit_after_gap(self, mean, covariance, measurement, gap_frames, score=None, aspect_scale=None):
        return reinit_after_gap(self, mean, covariance, measurement, gap_frames, score, aspect_scale)

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric="maha"):
        """Compute Gaussian or squared Mahalanobis gating distance."""
        projected_mean, projected_covariance = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_covariance = projected_covariance[:2, :2]
            measurements = measurements[:, :2]
        difference = measurements - projected_mean
        if metric == "gaussian":
            return np.sum(difference * difference, axis=1)
        if metric != "maha":
            raise ValueError("invalid distance metric")
        cholesky = np.linalg.cholesky(projected_covariance)
        normalized = scipy.linalg.solve_triangular(
            cholesky, difference.T, lower=True, check_finite=False,
            overwrite_b=True)
        return np.sum(normalized * normalized, axis=0)

    @property
    def score_weight_enabled(self):
        return self._enable_score_weight

    @property
    def gap_recovery_enabled(self):
        return self._enable_gap_recovery

    @property
    def aspect_adaptive_enabled(self):
        return self._enable_aspect_adaptive
