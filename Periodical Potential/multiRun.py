def run_model(model):
        model.run(50000)


if __name__ == "__main__":
    import numpy as np
    from itertools import product
    from main import PeriodicalPotential
    from multiprocessing import Pool

    rangeLambdas = np.concatenate([
        np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.2)
    ])
    distanceDs = np.concatenate([
        np.arange(0.1, 1, 0.2)
    ])
    rangeGamma = np.linspace(0, 2, 5)
    rangeKappa = np.linspace(0, 1, 5)
    rangePeriod = np.linspace(0.1, 2.5, 6)

    savePath = "./data"

    models = [
        PeriodicalPotential(
            strengthLambda=l, distanceD=d, gamma=g, kappa=k, L=period,
            agentsNum=1000, boundaryLength=5,
            tqdm=True, savePath=savePath, overWrite=False)
        for l, d, g, k, period in product(rangeLambdas, distanceDs, rangeGamma, rangeKappa, rangePeriod)
    ]

    with Pool(40) as p:
        p.map(run_model, models)
