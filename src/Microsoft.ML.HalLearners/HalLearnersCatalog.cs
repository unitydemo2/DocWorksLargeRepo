// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.HalLearners;
using Microsoft.ML.Trainers.SymSgd;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    ///     <summary>
        ///     The trainer context extensions for the <see cref="OlsLinearRegressionTrainer"/> and <see cref="SymSgdClassificationTrainer"/>.
        ///     </summary>
            public static class HalLearnersCatalog
    {
        ///     <summary>
                ///     Predict a target using a linear regression model trained with the <see cref="OlsLinearRegressionTrainer"/>.
                ///     </summary>
                ///     <param name="ctx">The <see cref="RegressionContext"/>.</param>
                ///     <param name="labelColumn">The labelColumn column.</param>
                ///     <param name="featureColumn">The features column.</param>
                ///     <param name="weights">The weights column.</param>
                ///     <param name="advancedSettings">Algorithm advanced settings.</param>
                        public static OlsLinearRegressionTrainer OrdinaryLeastSquares(this RegressionContext.RegressionTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string weights = null,
            Action<OlsLinearRegressionTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new OlsLinearRegressionTrainer(env, labelColumn, featureColumn, weights, advancedSettings);
        }

        ///     <summary>
                ///      Predict a target using a linear binary classification model trained with the <see cref="SymSgdClassificationTrainer"/>.
                ///     </summary>
                ///     <param name="ctx">The <see cref="BinaryClassificationContext"/>.</param>
                ///     <param name="labelColumn">The labelColumn column.</param>
                ///     <param name="featureColumn">The features column.</param>
                ///     <param name="advancedSettings">Algorithm advanced settings.</param>
                        public static SymSgdClassificationTrainer SymbolicStochasticGradientDescent(this BinaryClassificationContext.BinaryClassificationTrainers ctx,
            string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            Action<SymSgdClassificationTrainer.Arguments> advancedSettings = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            var env = CatalogUtils.GetEnvironment(ctx);
            return new SymSgdClassificationTrainer(env, labelColumn, featureColumn, advancedSettings);
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.HalLearnersCatalog.VectorWhiten(Microsoft.ML.TransformsCatalog.ProjectionTransforms,System.String,System.String,Microsoft.ML.Transforms.Projections.WhiteningKind,System.Single,System.Int32,System.Int32)" -->
                        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn = null,
            WhiteningKind kind = VectorWhiteningTransformer.Defaults.Kind,
            float eps = VectorWhiteningTransformer.Defaults.Eps,
            int maxRows = VectorWhiteningTransformer.Defaults.MaxRows,
            int pcaNum = VectorWhiteningTransformer.Defaults.PcaNum)
                => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, kind, eps, maxRows, pcaNum);

        ///     <summary>
                ///     Takes columns filled with a vector of random variables with a known covariance matrix into a set of new variables whose covariance is the identity matrix,
                ///     meaning that they are uncorrelated and each have variance 1.
                ///     </summary>
                ///     <param name="catalog">The transform's catalog.</param>
                ///     <param name="columns">Describes the parameters of the whitening process for each column pair.</param>
                        public static VectorWhiteningEstimator VectorWhiten(this TransformsCatalog.ProjectionTransforms catalog, params VectorWhiteningTransformer.ColumnInfo[] columns)
            => new VectorWhiteningEstimator(CatalogUtils.GetEnvironment(catalog), columns);

    }
}
