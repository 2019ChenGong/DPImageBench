from math import ceil

from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.transformer_zoo import AmplificationBySampling, Composition
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import PRVAccountant


def get_noisysgd_mechanism(noise_scale, sample_rate, max_steps):
    mech = ExactGaussianMechanism(sigma=noise_scale)
    # Support multiple versions of autodp by setting both of these fields
    mech.replace_one = True
    mech.neighboring = 'replace_one'

    subsample = AmplificationBySampling(PoissonSampling=False)
    SubsampledGaussian_mech = subsample(mech, sample_rate, improved_bound_flag=True)

    compose = Composition()
    Composed_mech = compose([SubsampledGaussian_mech], [max_steps])
    Composed_mech.params["noise_scale"] = noise_scale

    return Composed_mech


def compute_noise_multiplier_poisson(dp_config, sample_rate, max_epochs):
    return get_noise_multiplier(
        target_epsilon=dp_config["epsilon"],
        target_delta=dp_config["delta"],
        sample_rate=sample_rate,
        epochs=max_epochs,
        accountant="prv",
        epsilon_tolerance=1e-3
    )


def compute_noise_multiplier_swr(dp_config, sample_rate, max_epochs):
    max_steps = ceil(max_epochs / sample_rate)
    def calibration_fn(noise_scale):
        return get_noisysgd_mechanism(noise_scale, sample_rate, max_steps)
    calibrate = eps_delta_calibrator()
    mech = calibrate(calibration_fn, dp_config["epsilon"], dp_config["delta"], [0, 1000])
    return mech.params["noise_scale"]


def compute_noise_multiplier(dp_config, sample_rate, max_epochs):
    if dp_config is None or not dp_config["enabled"]:
        return 0
    elif dp_config["poisson_sampling"]:
        return compute_noise_multiplier_poisson(dp_config, sample_rate, max_epochs)
    else:
        # This computation uses the replacement definition of DP, while Opacus'
        # noise addition assumes the add/replace definition. We need to multiply
        # the noise scale by 2 to accomodate the doubled sensitivity.
        return 2 * compute_noise_multiplier_swr(dp_config, sample_rate, max_epochs)


if __name__ == "__main__":
    print("#### Tests ####")

    print("Sampling without replacement:")
    print(
        "  AutoDP (sigma=1, sample_rate=0.01, steps=100000, delta=1e-5) => epsilon =",
        get_noisysgd_mechanism(1, 0.01, 100000).get_approxDP(1e-5)
    )

    print("Poisson sampling:")
    mech = ExactGaussianMechanism(sigma=1)
    subsample = AmplificationBySampling(PoissonSampling=True)
    SubsampledGaussian_mech = subsample(mech, 0.01, improved_bound_flag=True)
    composed_gm = Composition()([SubsampledGaussian_mech], [100000])
    print(
        "  AutoDP (sigma=1, sample_rate=0.01, steps=100000, delta=1e-5) => epsilon =",
        composed_gm.get_approxDP(1e-5)
    )

    accountant = PRVAccountant()
    accountant.history = [(1, 0.01, 100000)]
    print(
        "  Opacus (sigma=1, sample_rate=0.01, steps=100000, delta=1e-5) => epsilon =",
        accountant.get_epsilon(delta=1e-5)
    )
