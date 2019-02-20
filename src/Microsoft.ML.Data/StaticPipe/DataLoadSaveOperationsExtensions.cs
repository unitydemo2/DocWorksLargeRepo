// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using static Microsoft.ML.Data.TextLoader;

namespace Microsoft.ML.StaticPipe
{
    
    public static class DataLoadSaveOperationsExtensions
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.DataLoadSaveOperationsExtensions.CreateTextReader``1(Microsoft.ML.DataOperations,System.Func{Microsoft.ML.Data.TextLoader.Context,``0},Microsoft.ML.Data.IMultiStreamSource,System.Boolean,System.Char,System.Boolean,System.Boolean,System.Boolean)" -->
                        public static DataReader<IMultiStreamSource, TShape> CreateTextReader<[IsShape] TShape>(
            this DataOperations catalog, Func<Context, TShape> func, IMultiStreamSource files = null,
            bool hasHeader = false, char separator = '\t', bool allowQuoting = true, bool allowSparse = true,
            bool trimWhitspace = false)
         => CreateReader(catalog.Environment, func, files, hasHeader, separator, allowQuoting, allowSparse, trimWhitspace);
    }
}
