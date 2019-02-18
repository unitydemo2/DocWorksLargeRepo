// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.FeatureSelection;

namespace Microsoft.ML
{
    using CountSelectDefaults = CountFeatureSelectingEstimator.Defaults;
    using MutualInfoSelectDefaults = MutualInformationFeatureSelectingEstimator.Defaults;

    
    public static class FeatureSelectionCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.FeatureSelectionCatalog.SelectFeaturesBasedOnMutualInformation(Microsoft.ML.TransformsCatalog.FeatureSelectionTransforms,System.String,System.Int32,System.Int32,System.ValueTuple{System.String,System.String}[])" -->
                        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string labelColumn = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numBins = MutualInfoSelectDefaults.NumBins,
            params (string input, string output)[] columns)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), labelColumn, slotsInOutput, numBins, columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.FeatureSelectionCatalog.SelectFeaturesBasedOnMutualInformation(Microsoft.ML.TransformsCatalog.FeatureSelectionTransforms,System.String,System.String,System.String,System.Int32,System.Int32)" -->
                        public static MutualInformationFeatureSelectingEstimator SelectFeaturesBasedOnMutualInformation(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string inputColumn, string outputColumn = null,
            string labelColumn = MutualInfoSelectDefaults.LabelColumn,
            int slotsInOutput = MutualInfoSelectDefaults.SlotsInOutput,
            int numBins = MutualInfoSelectDefaults.NumBins)
            => new MutualInformationFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, labelColumn, slotsInOutput, numBins);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.FeatureSelectionCatalog.SelectFeaturesBasedOnCount(Microsoft.ML.TransformsCatalog.FeatureSelectionTransforms,Microsoft.ML.Transforms.FeatureSelection.CountFeatureSelectingEstimator.ColumnInfo[])" -->
                        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            params CountFeatureSelectingEstimator.ColumnInfo[] columns)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.FeatureSelectionCatalog.SelectFeaturesBasedOnCount(Microsoft.ML.TransformsCatalog.FeatureSelectionTransforms,System.String,System.String,System.Int64)" -->
                        public static CountFeatureSelectingEstimator SelectFeaturesBasedOnCount(this TransformsCatalog.FeatureSelectionTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            long count = CountSelectDefaults.Count)
            => new CountFeatureSelectingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, count);
    }
}
