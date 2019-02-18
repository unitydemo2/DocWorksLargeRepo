// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Normalizers;

namespace Microsoft.ML
{
    ///     <summary>
        ///     Extensions for normalizer operations.
        ///     </summary>
            public static class NormalizerCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.NormalizerCatalog.Normalize(Microsoft.ML.TransformsCatalog,System.String,System.String,Microsoft.ML.Transforms.Normalizers.NormalizingEstimator.NormalizerMode)" -->
                        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            string inputName,
            string outputName = null,
            NormalizingEstimator.NormalizerMode mode = NormalizingEstimator.NormalizerMode.MinMax)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), inputName, outputName, mode);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.NormalizerCatalog.Normalize(Microsoft.ML.TransformsCatalog,Microsoft.ML.Transforms.Normalizers.NormalizingEstimator.NormalizerMode,System.ValueTuple{System.String,System.String}[])" -->
                        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            NormalizingEstimator.NormalizerMode mode,
            params (string input, string output)[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), mode, columns);

        ///     <summary>
                ///     Normalize (rescale) columns according to specified custom parameters.
                ///     </summary>
                ///     <param name="catalog">The transform catalog</param>
                ///     <param name="columns">The normalization settings for all the columns</param>
                        public static NormalizingEstimator Normalize(this TransformsCatalog catalog,
            params NormalizingEstimator.ColumnBase[] columns)
            => new NormalizingEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
