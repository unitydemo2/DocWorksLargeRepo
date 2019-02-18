// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Categorical;

namespace Microsoft.ML
{
    ///     <summary>
        ///     Static extensions for categorical transforms.
        ///     </summary>
            public static class CategoricalCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.CategoricalCatalog.OneHotEncoding(Microsoft.ML.TransformsCatalog.CategoricalTransforms,System.String,System.String,Microsoft.ML.Transforms.Categorical.OneHotEncodingTransformer.OutputKind)" -->
                        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn,
                string outputColumn = null,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, outputKind);

        ///     <summary>
                ///     Convert several text column into one-hot encoded vectors.
                ///     </summary>
                ///     <param name="catalog">The transform catalog</param>
                ///     <param name="columns">The column settings.</param>
                ///     <returns></returns>
                        public static OneHotEncodingEstimator OneHotEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotEncodingEstimator.ColumnInfo[] columns)
            => new OneHotEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.CategoricalCatalog.OneHotHashEncoding(Microsoft.ML.TransformsCatalog.CategoricalTransforms,System.String,System.String,System.Int32,System.Int32,Microsoft.ML.Transforms.Categorical.OneHotEncodingTransformer.OutputKind)" -->
                        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                string inputColumn,
                string outputColumn = null,
                int hashBits = OneHotHashEncodingEstimator.Defaults.HashBits,
                int invertHash = OneHotHashEncodingEstimator.Defaults.InvertHash,
                OneHotEncodingTransformer.OutputKind outputKind = OneHotEncodingTransformer.OutputKind.Ind)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, hashBits, invertHash, outputKind);

        ///     <summary>
                ///     Convert several text column into hash-based one-hot encoded vectors.
                ///     </summary>
                ///     <param name="catalog">The transform catalog</param>
                ///     <param name="columns">The column settings.</param>
                ///     <returns></returns>
                        public static OneHotHashEncodingEstimator OneHotHashEncoding(this TransformsCatalog.CategoricalTransforms catalog,
                params OneHotHashEncodingEstimator.ColumnInfo[] columns)
            => new OneHotHashEncodingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
