#!/usr/bin/env python

import sys
from math import sqrt
import itertools
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS


def parseRating(line):
    """
    Parses a rating record in MovieLens format
    userId::movieId::rating::timestamp .
    """
    fields = line.split("::")
    return (long(fields[3]) % 10,
            (int(fields[0]), int(fields[1]), float(fields[2]))
            )


def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.split("::")
    return int(fields[0]), fields[1]


def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings


def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = (
        predictions
        .map(lambda x: ((x[0], x[1]), x[2]))
        .join(data.map(lambda x: ((x[0], x[1]), x[2])))
        .values()
        )

    return sqrt(
        predictionsAndRatings
        .map(lambda x: (x[0] - x[1]) ** 2)
        .reduce(add) / float(n)
        )


if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # set up environment
    conf = (
        SparkConf()
        .setAppName("MovieLensALS")
        .set("spark.executor.memory", "2g")
        )
    sc = SparkContext(conf=conf)

    # Read in Ratings
    movieLensHomeDir = sys.argv[1]
    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = (
        sc
        .textFile(join(movieLensHomeDir, "ratings.dat"))
        .map(parseRating)
        )

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(
        sc
        .textFile(join(movieLensHomeDir, "movies.dat"))
        .map(parseMovie).collect()
        )

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    print 'Got {} ratings from {} users on {} movies.'.format(
        numRatings, numUsers, numMovies)

    numPartitions = 4
    training = (
        ratings
        .filter(lambda x: x[0] < 6)
        .values()
        .repartition(numPartitions)
        .cache()
        )

    validation = (
        ratings.filter(lambda x: x[0] >= 6 and x[0] < 8)
        .values()
        .repartition(numPartitions)
        .cache()
        )

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print 'Training: {}, validation: {}, test: {}'.format(
        numTraining, numValidation, numTest)

    # ranks = [8, 12]
    # lambdas = [1.0, 10.0]
    # numIters = [10, 20]
    ranks = [8]
    lambdas = [1.0]
    numIters = [10]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(training, rank, numIter, lmbda)
        validationRmse = computeRmse(model, validation, numValidation)
        print ('RMSE (validation) = {} for the model trained with '
               'rank = {}, lambda = {}, and numIter = {}.'
               .format(validationRmse, rank, lmbda, numIter))

        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print ('The best model was trained with rank = {}, lambda = {}, '
           'numIter = {}, and its RMSE on the test set is {}.'
           .format(bestRank, bestLambda, bestNumIter, testRmse))

    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(
        test
        .map(lambda x: (meanRating - x[2]) ** 2)
        .reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print ('The best model improves the baseline by {:0.2f}% '
           'where baseline RMSE = {:0.2f} and Best Model RMSE = {:0.2f}'
           .format(improvement, baselineRmse, testRmse))
    # load personal ratings
    # myRatings = loadRatings(sys.argv[2])
    # myRatingsRDD = sc.parallelize(myRatings, 1)

    # clean up
    sc.stop()
