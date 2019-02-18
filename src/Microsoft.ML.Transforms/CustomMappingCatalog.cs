// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    ///     <summary>
        ///     Extension methods for custom mapping transformers.
        ///     </summary>
            public static class CustomMappingCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.CustomMappingCatalog.CustomMapping``2(Microsoft.ML.TransformsCatalog,System.Action{``0,``1},System.String,Microsoft.ML.Data.SchemaDefinition,Microsoft.ML.Data.SchemaDefinition)" -->
                        public static CustomMappingEstimator<TSrc, TDst> CustomMapping<TSrc, TDst>(this TransformsCatalog catalog, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            => new CustomMappingEstimator<TSrc, TDst>(catalog.GetEnvironment(), mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.CustomMappingCatalog.CustomMappingTransformer``2(Microsoft.ML.TransformsCatalog,System.Action{``0,``1},System.String,Microsoft.ML.Data.SchemaDefinition,Microsoft.ML.Data.SchemaDefinition)" -->
                        public static CustomMappingTransformer<TSrc, TDst> CustomMappingTransformer<TSrc, TDst>(this TransformsCatalog catalog, Action<TSrc, TDst> mapAction, string contractName,
                SchemaDefinition inputSchemaDefinition = null, SchemaDefinition outputSchemaDefinition = null)
            where TSrc : class, new()
            where TDst : class, new()
            => new CustomMappingTransformer<TSrc, TDst>(catalog.GetEnvironment(), mapAction, contractName, inputSchemaDefinition, outputSchemaDefinition);
    }
}
