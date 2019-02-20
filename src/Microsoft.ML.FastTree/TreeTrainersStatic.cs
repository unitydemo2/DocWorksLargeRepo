// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.StaticPipe
{
    ///     <summary>
        ///     FastTree <see cref="TrainContextBase"/> extension methods.
        ///     </summary>
            public static class TreeRegressionExtensions
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.TreeRegressionExtensions.FastTree(Microsoft.ML.RegressionContext.RegressionTrainers,Microsoft.ML.StaticPipe.Scalar{System.Single},Microsoft.ML.StaticPipe.Vector{System.Single},Microsoft.ML.StaticPipe.Scalar{System.Single},System.Int32,System.Int32,System.Int32,System.Double,System.Action{Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Arguments},System.Action{Microsoft.ML.Trainers.FastTree.FastTreeRegressionModelParameters})" -->
                        public static Scalar<float> FastTree(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeRegressionTrainer.Arguments> advancedSettings = null,
            Action<FastTreeRegressionModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.Regression(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeRegressionTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, weights);

            return rec.Score;
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.TreeRegressionExtensions.FastTree(Microsoft.ML.BinaryClassificationContext.BinaryClassificationTrainers,Microsoft.ML.StaticPipe.Scalar{System.Boolean},Microsoft.ML.StaticPipe.Vector{System.Single},Microsoft.ML.StaticPipe.Scalar{System.Single},System.Int32,System.Int32,System.Int32,System.Double,System.Action{Microsoft.ML.Trainers.FastTree.FastTreeBinaryClassificationTrainer.Arguments},System.Action{Microsoft.ML.Internal.Internallearn.IPredictorWithFeatureWeights{System.Single}})" -->
                        public static (Scalar<float> score, Scalar<float> probability, Scalar<bool> predictedLabel) FastTree(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            Scalar<bool> label, Vector<float> features, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeBinaryClassificationTrainer.Arguments> advancedSettings = null,
            Action<IPredictorWithFeatureWeights<float>> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.BinaryClassifier(
               (env, labelName, featuresName, weightsName) =>
               {
                   var trainer = new FastTreeBinaryClassificationTrainer(env, labelName, featuresName, weightsName, numLeaves,
                       numTrees, minDatapointsInLeaves, learningRate, advancedSettings);

                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   else
                       return trainer;
               }, label, features, weights);

            return rec.Output;
        }

        ///     <summary>
               ///     FastTree <see cref="RankingContext"/>.
               ///     Ranks a series of inputs based on their relevance, training a decision tree ranking model through the <see cref="FastTreeRankingTrainer"/>.
               ///     </summary>
               ///     <param name="ctx">The <see cref="RegressionContext"/>.</param>
               ///     <param name="label">The label column.</param>
               ///     <param name="features">The features column.</param>
               ///     <param name="groupId">The groupId column.</param>
               ///     <param name="weights">The optional weights column.</param>
               ///     <param name="numTrees">Total number of decision trees to create in the ensemble.</param>
               ///     <param name="numLeaves">The maximum number of leaves per decision tree.</param>
               ///     <param name="minDatapointsInLeaves">The minimal number of datapoints allowed in a leaf of a regression tree, out of the subsampled data.</param>
               ///     <param name="learningRate">The learning rate.</param>
               ///     <param name="advancedSettings">Algorithm advanced settings.</param>
               ///     <param name="onFit">A delegate that is called every time the
               ///     <see cref="Estimator{TInShape, TOutShape, TTransformer}.Fit(DataView{TInShape})"/> method is called on the
               ///     <see cref="Estimator{TInShape, TOutShape, TTransformer}"/> instance created out of this. This delegate will receive
               ///     the linear model that was trained. Note that this action cannot change the result in any way;
               ///     it is only a way for the caller to be informed about what was learnt.</param>
               ///     <returns>The Score output column indicating the predicted value.</returns>
                      public static Scalar<float> FastTree<TVal>(this RankingContext.RankingTrainers ctx,
            Scalar<float> label, Vector<float> features, Key<uint, TVal> groupId, Scalar<float> weights = null,
            int numLeaves = Defaults.NumLeaves,
            int numTrees = Defaults.NumTrees,
            int minDatapointsInLeaves = Defaults.MinDocumentsInLeaves,
            double learningRate = Defaults.LearningRates,
            Action<FastTreeRankingTrainer.Arguments> advancedSettings = null,
            Action<FastTreeRankingModelParameters> onFit = null)
        {
            CheckUserValues(label, features, weights, numLeaves, numTrees, minDatapointsInLeaves, learningRate, advancedSettings, onFit);

            var rec = new TrainerEstimatorReconciler.Ranker<TVal>(
               (env, labelName, featuresName, groupIdName, weightsName) =>
               {
                   var trainer = new FastTreeRankingTrainer(env, labelName, featuresName, groupIdName, weightsName, numLeaves,
                       numTrees, minDatapointsInLeaves, learningRate, advancedSettings);
                   if (onFit != null)
                       return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                   return trainer;
               }, label, features, groupId, weights);

            return rec.Score;
        }

        internal static void CheckUserValues(PipelineColumn label, Vector<float> features, Scalar<float> weights,
            int numLeaves,
            int numTrees,
            int minDatapointsInLeaves,
            double learningRate,
            Delegate advancedSettings,
            Delegate onFit)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(numLeaves >= 2, nameof(numLeaves), "Must be at least 2.");
            Contracts.CheckParam(numTrees > 0, nameof(numTrees), "Must be positive");
            Contracts.CheckParam(minDatapointsInLeaves > 0, nameof(minDatapointsInLeaves), "Must be positive");
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive");
            Contracts.CheckValueOrNull(advancedSettings);
            Contracts.CheckValueOrNull(onFit);
        }
    }
}
