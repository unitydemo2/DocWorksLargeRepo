// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Learners;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Trainers.Online;

namespace Microsoft.ML.StaticPipe
{
    ///     <summary>
        ///     Binary Classification trainer estimators.
        ///     </summary>
            public static class AveragedPerceptronExtensions
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.AveragedPerceptronExtensions.AveragedPerceptron(Microsoft.ML.BinaryClassificationContext.BinaryClassificationTrainers,Microsoft.ML.StaticPipe.Scalar{System.Boolean},Microsoft.ML.StaticPipe.Vector{System.Single},Microsoft.ML.StaticPipe.Scalar{System.Single},Microsoft.ML.IClassificationLoss,System.Single,System.Boolean,System.Single,System.Int32,System.Action{Microsoft.ML.Trainers.Online.AveragedPerceptronTrainer.Arguments},System.Action{Microsoft.ML.Learners.LinearBinaryModelParameters})" -->
                        public static (Scalar<float> score, Scalar<bool> predictedLabel) AveragedPerceptron(
                this BinaryClassificationContext.BinaryClassificationTrainers ctx,
                Scalar<bool> label,
                Vector<float> features,
                Scalar<float> weights = null,
                IClassificationLoss lossFunction = null,
                float learningRate = AveragedLinearArguments.AveragedDefaultArgs.LearningRate,
                bool decreaseLearningRate = AveragedLinearArguments.AveragedDefaultArgs.DecreaseLearningRate,
                float l2RegularizerWeight = AveragedLinearArguments.AveragedDefaultArgs.L2RegularizerWeight,
                int numIterations = AveragedLinearArguments.AveragedDefaultArgs.NumIterations,
                Action<AveragedPerceptronTrainer.Arguments> advancedSettings = null,
                Action<LinearBinaryModelParameters> onFit = null
            )
        {
            OnlineLinearStaticUtils.CheckUserParams(label, features, weights, learningRate, l2RegularizerWeight, numIterations, onFit, advancedSettings);

            bool hasProbs = lossFunction is LogLoss;

            var rec = new TrainerEstimatorReconciler.BinaryClassifierNoCalibration(
                (env, labelName, featuresName, weightsName) =>
                {

                    var trainer = new AveragedPerceptronTrainer(env, labelName, featuresName, weightsName, lossFunction,
                        learningRate, decreaseLearningRate, l2RegularizerWeight, numIterations, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));
                    else
                        return trainer;

                }, label, features, weights, hasProbs);

            return rec.Output;
        }
    }

    ///     <summary>
        ///     Regression trainer estimators.
        ///     </summary>
            public static class OnlineGradientDescentExtensions
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.OnlineGradientDescentExtensions.OnlineGradientDescent(Microsoft.ML.RegressionContext.RegressionTrainers,Microsoft.ML.StaticPipe.Scalar{System.Single},Microsoft.ML.StaticPipe.Vector{System.Single},Microsoft.ML.StaticPipe.Scalar{System.Single},Microsoft.ML.IRegressionLoss,System.Single,System.Boolean,System.Single,System.Int32,System.Action{Microsoft.ML.Trainers.Online.AveragedLinearArguments},System.Action{Microsoft.ML.Learners.LinearRegressionModelParameters})" -->
                        public static Scalar<float> OnlineGradientDescent(this RegressionContext.RegressionTrainers ctx,
            Scalar<float> label,
            Vector<float> features,
            Scalar<float> weights = null,
            IRegressionLoss lossFunction = null,
            float learningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.LearningRate,
            bool decreaseLearningRate = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.DecreaseLearningRate,
            float l2RegularizerWeight = OnlineGradientDescentTrainer.Arguments.OgdDefaultArgs.L2RegularizerWeight,
            int numIterations = OnlineLinearArguments.OnlineDefaultArgs.NumIterations,
            Action<AveragedLinearArguments> advancedSettings = null,
            Action<LinearRegressionModelParameters> onFit = null)
        {
            OnlineLinearStaticUtils.CheckUserParams(label, features, weights, learningRate, l2RegularizerWeight, numIterations, onFit, advancedSettings);
            Contracts.CheckValueOrNull(lossFunction);

            var rec = new TrainerEstimatorReconciler.Regression(
                (env, labelName, featuresName, weightsName) =>
                {
                    var trainer = new OnlineGradientDescentTrainer(env, labelName, featuresName, learningRate,
                        decreaseLearningRate, l2RegularizerWeight, numIterations, weightsName, lossFunction, advancedSettings);

                    if (onFit != null)
                        return trainer.WithOnFitDelegate(trans => onFit(trans.Model));

                    return trainer;
                }, label, features, weights);

            return rec.Score;
        }
    }

    internal static class OnlineLinearStaticUtils{

        internal static void CheckUserParams(PipelineColumn label,
            PipelineColumn features,
            PipelineColumn weights,
            float learningRate,
            float l2RegularizerWeight,
            int numIterations,
            Delegate onFit,
            Delegate advancedArguments)
        {
            Contracts.CheckValue(label, nameof(label));
            Contracts.CheckValue(features, nameof(features));
            Contracts.CheckValueOrNull(weights);
            Contracts.CheckParam(learningRate > 0, nameof(learningRate), "Must be positive.");
            Contracts.CheckParam(0 <= l2RegularizerWeight && l2RegularizerWeight < 0.5, nameof(l2RegularizerWeight), "must be in range [0, 0.5)");
            Contracts.CheckParam(numIterations > 0, nameof(numIterations), "Must be positive, if specified.");
            Contracts.CheckValueOrNull(onFit);
            Contracts.CheckValueOrNull(advancedArguments);
        }
    }
}
