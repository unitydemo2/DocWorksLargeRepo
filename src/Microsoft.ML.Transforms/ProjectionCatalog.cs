// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Projections;

namespace Microsoft.ML
{
    
    public static class ProjectionCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.ProjectionCatalog.CreateRandomFourierFeatures(Microsoft.ML.TransformsCatalog.ProjectionTransforms,System.String,System.String,System.Int32,System.Boolean)" -->
                        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int newDim = RandomFourierFeaturizingEstimator.Defaults.NewDim,
            bool useSin = RandomFourierFeaturizingEstimator.Defaults.UseSin)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, newDim, useSin);

        ///     <summary>
                ///     Takes columns filled with a vector of floats and maps its to a random low-dimensional feature space.
                ///     </summary>
                ///     <param name="catalog">The transform's catalog.</param>
                ///     <param name="columns">The input columns to use for the transformation.</param>
                        public static RandomFourierFeaturizingEstimator CreateRandomFourierFeatures(this TransformsCatalog.ProjectionTransforms catalog, params RandomFourierFeaturizingTransformer.ColumnInfo[] columns)
            => new RandomFourierFeaturizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.ProjectionCatalog.LpNormalize(Microsoft.ML.TransformsCatalog.ProjectionTransforms,System.String,System.String,Microsoft.ML.Transforms.Projections.LpNormalizingEstimatorBase.NormalizerKind,System.Boolean)" -->
                        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn =null,
            LpNormalizingEstimatorBase.NormalizerKind normKind = LpNormalizingEstimatorBase.Defaults.NormKind, bool subMean = LpNormalizingEstimatorBase.Defaults.LpSubstractMean)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, normKind, subMean);

        ///     <summary>
                ///     Takes columns filled with a vector of floats and computes L-p norm of it.
                ///     </summary>
                ///     <param name="catalog">The transform's catalog.</param>
                ///     <param name="columns"> Describes the parameters of the lp-normalization process for each column pair.</param>
                        public static LpNormalizingEstimator LpNormalize(this TransformsCatalog.ProjectionTransforms catalog, params LpNormalizingTransformer.LpNormColumnInfo[] columns)
            => new LpNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.ProjectionCatalog.GlobalContrastNormalize(Microsoft.ML.TransformsCatalog.ProjectionTransforms,System.String,System.String,System.Boolean,System.Boolean,System.Single)" -->
                        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, string inputColumn, string outputColumn = null,
             bool substractMean = LpNormalizingEstimatorBase.Defaults.GcnSubstractMean,
             bool useStdDev = LpNormalizingEstimatorBase.Defaults.UseStdDev,
             float scale = LpNormalizingEstimatorBase.Defaults.Scale)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, substractMean, useStdDev, scale);

        ///     <summary>
                ///     Takes columns filled with a vector of floats and computes global contrast normalization of it.
                ///     </summary>
                ///     <param name="catalog">The transform's catalog.</param>
                ///     <param name="columns"> Describes the parameters of the gcn-normaliztion process for each column pair.</param>
                        public static GlobalContrastNormalizingEstimator GlobalContrastNormalize(this TransformsCatalog.ProjectionTransforms catalog, params LpNormalizingTransformer.GcnColumnInfo[] columns)
            => new GlobalContrastNormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
