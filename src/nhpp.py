import numpy as np
import random
import utils


class NHPP:

    def __init__(self, intensity_func, critical_points: list, seed: int = 0):

        self.__intensity_func = intensity_func
        self.__critical_points = critical_points
        self.__seed = seed

        self.__init_time = self.__critical_points[0]
        self.__numOfTimeBins = len(self.__critical_points)
        # Find the max lambda values for each interval
        self.__lambdaValues = [0]
        for idx in range(1, self.__numOfTimeBins):
            self.__lambdaValues.append(
                max(self.__intensity_func(t=self.__critical_points[idx - 1]), intensity_func(t=self.__critical_points[idx]))
            )

        # Set seed
        random.seed(self.__seed)
        np.random.seed(self.__seed)

    def simulate(self) -> list:
        t, J, S = self.__init_time, 1, []
        # Step 2
        U = np.random.uniform(low=0, high=1) # Random number
        X = (-1.0/(self.__lambdaValues[J]+utils.EPS)) * np.log(U) # Random variable from exponential dist for NHPP time step

        while True:
            # Step 3
            if t + X < self.__critical_points[J]:
                # Step 4
                t = t + X
                # Step 5
                U = np.random.uniform(low=0, high=1)
                # Step 6
                if U <= self.__intensity_func(t)/self.__lambdaValues[J]:
                    # Don't need I for index, because we append t to S
                    S.append(t)
                # Step 7 -> Do step 2 then loop starts again at step 3
                U = np.random.uniform(low=0, high=1)  # Random number
                X = (-1./self.__lambdaValues[J]) * np.log(U)  # Random variable from exponential dist for NHPP time step
            else:
                # Step 8
                if J == self.__numOfTimeBins - 1: #k +1 because zero-indexing
                    break
                # Step 9
                X = (X-self.__critical_points[J] + t) * self.__lambdaValues[J]/self.__lambdaValues[J+1]
                t = self.__critical_points[J]
                J += 1

        return S