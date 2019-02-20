// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data
{
    ///     <summary>
        ///     Utilities for implementing and using the metadata API of <see cref="Schema"/>.
        ///     </summary>
            public static class MetadataUtils
    {
        /// <summary>
        /// This class lists the canonical metadata kinds
        /// </summary>
        public static class Kinds
        {
            /// <summary>
            /// Metadata kind for names associated with slots/positions in a vector-valued column.
            /// The associated metadata type is typically fixed-sized vector of Text.
            /// </summary>
            public const string SlotNames = "SlotNames";

            /// <summary>
            /// Metadata kind for values associated with the key indices when the column type's item type
            /// is a key type. The associated metadata type is typically fixed-sized vector of a primitive
            /// type. The primitive type is frequently Text, but can be anything.
            /// </summary>
            public const string KeyValues = "KeyValues";

            /// <summary>
            /// Metadata kind for sets of score columns. The value is typically a KeyType with raw type U4.
            /// </summary>
            public const string ScoreColumnSetId = "ScoreColumnSetId";

            /// <summary>
            /// Metadata kind that indicates the prediction kind as a string. For example, "BinaryClassification".
            /// The value is typically a ReadOnlyMemory<char>.
            /// </summary>
            public const string ScoreColumnKind = "ScoreColumnKind";

            /// <summary>
            /// Metadata kind that indicates the value kind of the score column as a string. For example, "Score", "PredictedLabel", "Probability". The value is typically a ReadOnlyMemory.
            /// </summary>
            public const string ScoreValueKind = "ScoreValueKind";

            /// <summary>
            /// Metadata kind that indicates if a column is normalized. The value is typically a Bool.
            /// </summary>
            public const string IsNormalized = "IsNormalized";

            /// <summary>
            /// Metadata kind that indicates if a column is visible to the users. The value is typically a Bool.
            /// Not to be confused with IsHidden() that determines if a column is masked.
            /// </summary>
            public const string IsUserVisible = "IsUserVisible";

            /// <summary>
            /// Metadata kind for the label values used in training to be used for the predicted label.
            /// The value is typically a fixed-sized vector of Text.
            /// </summary>
            public const string TrainingLabelValues = "TrainingLabelValues";

            /// <summary>
            /// Metadata kind that indicates the ranges within a column that are categorical features.
            /// The value is a vector type of ints with dimension of two. The first dimension
            /// represents the number of categorical features and second dimension represents the range
            /// and is of size two. The range has start and end index(both inclusive) of categorical
            /// slots within that column.
            /// </summary>
            public const string CategoricalSlotRanges = "CategoricalSlotRanges";
        }

        /// <summary>
        /// This class holds all pre-defined string values that can be found in canonical metadata
        /// </summary>
        public static class Const
        {
            
            public static class ScoreColumnKind
            {
                
                public const string BinaryClassification = "BinaryClassification";
                
                public const string MultiClassClassification = "MultiClassClassification";
                
                public const string Regression = "Regression";
                
                public const string Ranking = "Ranking";
                
                public const string Clustering = "Clustering";
                
                public const string MultiOutputRegression = "MultiOutputRegression";
                
                public const string AnomalyDetection = "AnomalyDetection";
                
                public const string SequenceClassification = "SequenceClassification";
                
                public const string QuantileRegression = "QuantileRegression";
                
                public const string Recommender = "Recommender";
                
                public const string ItemSimilarity = "ItemSimilarity";
                
                public const string FeatureContribution = "FeatureContribution";
            }

            
            public static class ScoreValueKind
            {
                
                public const string Score = "Score";
                
                public const string PredictedLabel = "PredictedLabel";
                
                public const string Probability = "Probability";
            }
        }

        /// <summary>
        /// Helper delegate for marshaling from generic land to specific types. Used by the Marshal method below.
        /// </summary>
        public delegate void MetadataGetter<TValue>(int col, ref TValue dst);

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetMetadata.
        /// </summary>
        [BestFriend]
        internal static Exception ExceptGetMetadata() => Contracts.Except("Invalid call to GetMetadata");

        /// <summary>
        /// Returns a standard exception for responding to an invalid call to GetMetadata.
        /// </summary>
        [BestFriend]
        internal static Exception ExceptGetMetadata(this IExceptionContext ctx) => ctx.Except("Invalid call to GetMetadata");

        /// <summary>
        /// Helper to marshal a call to GetMetadata{TValue} to a specific type.
        /// </summary>
        [BestFriend]
        internal static void Marshal<THave, TNeed>(this MetadataGetter<THave> getter, int col, ref TNeed dst)
        {
            Contracts.CheckValue(getter, nameof(getter));

            if (typeof(TNeed) != typeof(THave))
                throw ExceptGetMetadata();
            var get = (MetadataGetter<TNeed>)(Delegate)getter;
            get(col, ref dst);
        }

        /// <summary>
        /// Returns a vector type with item type text and the given size. The size must be positive.
        /// This is a standard type for metadata consisting of multiple text values, eg SlotNames.
        /// </summary>
        [BestFriend]
        internal static VectorType GetNamesType(int size)
        {
            Contracts.CheckParam(size > 0, nameof(size), "must be known size");
            return new VectorType(TextType.Instance, size);
        }

        /// <summary>
        /// Returns a vector type with item type int and the given size.
        /// The range count must be a positive integer.
        /// This is a standard type for metadata consisting of multiple int values that represent
        /// categorical slot ranges with in a column.
        /// </summary>
        [BestFriend]
        internal static VectorType GetCategoricalType(int rangeCount)
        {
            Contracts.CheckParam(rangeCount > 0, nameof(rangeCount), "must be known size");
            return new VectorType(NumberType.I4, rangeCount, 2);
        }

        private static volatile KeyType _scoreColumnSetIdType;

        /// <summary>
        /// The type of the ScoreColumnSetId metadata.
        /// </summary>
        [BestFriend]
        internal static KeyType ScoreColumnSetIdType
        {
            get
            {
                if (_scoreColumnSetIdType == null)
                {
                    var type = new KeyType(DataKind.U4, 0, 0);
                    Interlocked.CompareExchange(ref _scoreColumnSetIdType, type, null);
                }
                return _scoreColumnSetIdType;
            }
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetMetadataTypes(col).
        /// </summary>
        [BestFriend]
        internal static KeyValuePair<string, ColumnType> GetSlotNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.SlotNames);
        }

        /// <summary>
        /// Returns a key-value pair useful when implementing GetMetadataTypes(col). This assumes
        /// that the values of the key type are Text.
        /// </summary>
        [BestFriend]
        internal static KeyValuePair<string, ColumnType> GetKeyNamesPair(int size)
        {
            return GetNamesType(size).GetPair(Kinds.KeyValues);
        }

        /// <summary>
        /// Given a type and metadata kind string, returns a key-value pair. This is useful when
        /// implementing GetMetadataTypes(col).
        /// </summary>
        [BestFriend]
        internal static KeyValuePair<string, ColumnType> GetPair(this ColumnType type, string kind)
        {
            Contracts.CheckValue(type, nameof(type));
            return new KeyValuePair<string, ColumnType>(kind, type);
        }

        // REVIEW: This should be in some general utility code.

        /// <summary>
        /// Prepends a params array to an enumerable. Useful when implementing GetMetadataTypes.
        /// </summary>
        [BestFriend]
        internal static IEnumerable<T> Prepend<T>(this IEnumerable<T> tail, params T[] head)
        {
            return head.Concat(tail);
        }

        ///     <summary>
                ///     Returns the max value for the specified metadata kind.
                ///     The metadata type should be a KeyType with raw type U4.
                ///     colMax will be set to the first column that has the max value for the specified metadata.
                ///     If no column has the specified metadata, colMax is set to -1 and the method returns zero.
                ///     The filter function is called for each column, passing in the schema and the column index, and returns
                ///     true if the column should be considered, false if the column should be skipped.
                ///     </summary>
                        public static uint GetMaxMetadataKind(this Schema schema, out int colMax, string metadataKind, Func<Schema, int, bool> filterFunc = null)
        {
            uint max = 0;
            colMax = -1;
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Metadata.Schema.GetColumnOrNull(metadataKind)?.Type;
                if (columnType == null || !columnType.IsKey || columnType.RawKind != DataKind.U4)
                    continue;
                if (filterFunc != null && !filterFunc(schema, col))
                    continue;
                uint value = 0;
                schema[col].Metadata.GetValue(metadataKind, ref value);
                if (max < value)
                {
                    max = value;
                    colMax = col;
                }
            }
            return max;
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified metadata kind.
        /// The metadata type should be a KeyType with raw type U4.
        /// </summary>
        [BestFriend]
        internal static IEnumerable<int> GetColumnSet(this Schema schema, string metadataKind, uint value)
        {
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Metadata.Schema.GetColumnOrNull(metadataKind)?.Type;
                if (columnType != null && columnType.IsKey && columnType.RawKind == DataKind.U4)
                {
                    uint val = 0;
                    schema[col].Metadata.GetValue(metadataKind, ref val);
                    if (val == value)
                        yield return col;
                }
            }
        }

        /// <summary>
        /// Returns the set of column ids which match the value of specified metadata kind.
        /// The metadata type should be of type text.
        /// </summary>
        [BestFriend]
        internal static IEnumerable<int> GetColumnSet(this Schema schema, string metadataKind, string value)
        {
            for (int col = 0; col < schema.Count; col++)
            {
                var columnType = schema[col].Metadata.Schema.GetColumnOrNull(metadataKind)?.Type;
                if (columnType != null && columnType.IsText)
                {
                    ReadOnlyMemory<char> val = default;
                    schema[col].Metadata.GetValue(metadataKind, ref val);
                    if (ReadOnlyMemoryUtils.EqualsStr(value, val))
                        yield return col;
                }
            }
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Data.MetadataUtils.HasSlotNames(Microsoft.ML.Data.Schema.Column)" -->
                        public static bool HasSlotNames(this Schema.Column column)
            => column.Type.IsKnownSizeVector && column.HasSlotNames(column.Type.VectorSize);

        /// <summary>
        /// Returns <c>true</c> if the specified column:
        ///  * has a SlotNames metadata
        ///  * metadata type is VBuffer<ReadOnlyMemory<char>> of length <paramref name="vectorSize"/>.
        /// </summary>
        [BestFriend]
        internal static bool HasSlotNames(this Schema.Column column, int vectorSize)
        {
            if (vectorSize == 0)
                return false;

            var metaColumn = column.Metadata.Schema.GetColumnOrNull(Kinds.SlotNames);
            return
                metaColumn != null
                && metaColumn.Value.Type.IsVector
                && metaColumn.Value.Type.VectorSize == vectorSize
                && metaColumn.Value.Type.ItemType.IsText;
        }

        
        public static void GetSlotNames(this Schema.Column column, ref VBuffer<ReadOnlyMemory<char>> slotNames)
            => column.Metadata.GetValue(Kinds.SlotNames, ref slotNames);

        [BestFriend]
        internal static void GetSlotNames(RoleMappedSchema schema, RoleMappedSchema.ColumnRole role, int vectorSize, ref VBuffer<ReadOnlyMemory<char>> slotNames)
        {
            Contracts.CheckValueOrNull(schema);
            Contracts.CheckParam(vectorSize >= 0, nameof(vectorSize));

            IReadOnlyList<Schema.Column> list = schema?.GetColumns(role);
            if (list?.Count != 1 || !schema.Schema[list[0].Index].HasSlotNames(vectorSize))
                VBufferUtils.Resize(ref slotNames, vectorSize, 0);
            else
                schema.Schema[list[0].Index].Metadata.GetValue(Kinds.SlotNames, ref slotNames);
        }

        [BestFriend]
        internal static bool HasKeyValues(this Schema.Column column, int keyCount)
        {
            if (keyCount == 0)
                return false;

            var metaColumn = column.Metadata.Schema.GetColumnOrNull(Kinds.KeyValues);
            return
                metaColumn != null
                && metaColumn.Value.Type.IsVector
                && metaColumn.Value.Type.VectorSize == keyCount
                && metaColumn.Value.Type.ItemType.IsText;
        }

        [BestFriend]
        internal static bool HasKeyValues(this SchemaShape.Column col)
        {
            return col.Metadata.TryFindColumn(Kinds.KeyValues, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Vector
                && metaCol.ItemType.IsText;
        }

        ///     <summary>
                ///     Returns true iff <paramref name="column"/> has IsNormalized metadata set to true.
                ///     </summary>
                        public static bool IsNormalized(this Schema.Column column)
        {
            var metaColumn = column.Metadata.Schema.GetColumnOrNull((Kinds.IsNormalized));
            if (metaColumn == null || !metaColumn.Value.Type.IsBool)
                return false;

            bool value = default;
            column.Metadata.GetValue(Kinds.IsNormalized, ref value);
            return value;
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Data.MetadataUtils.IsNormalized(Microsoft.ML.Core.Data.SchemaShape.Column)" -->
                        public static bool IsNormalized(this SchemaShape.Column col)
        {
            Contracts.CheckParam(col.IsValid, nameof(col), "struct not initialized properly");
            return col.Metadata.TryFindColumn(Kinds.IsNormalized, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Scalar && !metaCol.IsKey
                && metaCol.ItemType == BoolType.Instance;
        }

        ///     <summary>
                ///     Returns whether a column has the <see cref="Kinds.SlotNames"/> metadata indicated by
                ///     the schema shape.
                ///     </summary>
                ///     <param name="col">The schema shape column to query</param>
                ///     <returns>True if and only if the column is a definite sized vector type, has the
                ///     <see cref="Kinds.SlotNames"/> metadata of definite sized vectors of text.</returns>
                        public static bool HasSlotNames(this SchemaShape.Column col)
        {
            Contracts.CheckParam(col.IsValid, nameof(col), "struct not initialized properly");
            return col.Kind == SchemaShape.Column.VectorKind.Vector
                && col.Metadata.TryFindColumn(Kinds.SlotNames, out var metaCol)
                && metaCol.Kind == SchemaShape.Column.VectorKind.Vector && !metaCol.IsKey
                && metaCol.ItemType == TextType.Instance;
        }

        /// <summary>
        /// Tries to get the metadata kind of the specified type for a column.
        /// </summary>
        /// <typeparam name="T">The raw type of the metadata, should match the PrimitiveType type</typeparam>
        /// <param name="schema">The schema</param>
        /// <param name="type">The type of the metadata</param>
        /// <param name="kind">The metadata kind</param>
        /// <param name="col">The column</param>
        /// <param name="value">The value to return, if successful</param>
        /// <returns>True if the metadata of the right type exists, false otherwise</returns>
        [BestFriend]
        internal static bool TryGetMetadata<T>(this Schema schema, PrimitiveType type, string kind, int col, ref T value)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckValue(type, nameof(type));

            var metadataType = schema[col].Metadata.Schema.GetColumnOrNull(kind)?.Type;
            if (!type.Equals(metadataType))
                return false;
            schema[col].Metadata.GetValue(kind, ref value);
            return true;
        }

        /// <summary>
        /// The categoricalFeatures is a vector of the indices of categorical features slots.
        /// This vector should always have an even number of elements, and the elements should be parsed in groups of two consecutive numbers.
        /// So if its value is the range of numbers: 0,2,3,4,8,9
        /// look at it as [0,2],[3,4],[8,9].
        /// The way to interpret that is: feature with indices 0, 1, and 2 are one categorical
        /// Features with indices 3 and 4 are another categorical. Features 5 and 6 don't appear there, so they are not categoricals.
        /// </summary>
        [BestFriend]
        internal static bool TryGetCategoricalFeatureIndices(Schema schema, int colIndex, out int[] categoricalFeatures)
        {
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(colIndex >= 0, nameof(colIndex));

            bool isValid = false;
            categoricalFeatures = null;
            if (!(schema[colIndex].Type is VectorType vecType && vecType.Size > 0))
                return isValid;

            var type = schema[colIndex].Metadata.Schema.GetColumnOrNull(Kinds.CategoricalSlotRanges)?.Type;
            if (type?.RawType == typeof(VBuffer<int>))
            {
                VBuffer<int> catIndices = default(VBuffer<int>);
                schema[colIndex].Metadata.GetValue(Kinds.CategoricalSlotRanges, ref catIndices);
                VBufferUtils.Densify(ref catIndices);
                int columnSlotsCount = vecType.Size;
                if (catIndices.Length > 0 && catIndices.Length % 2 == 0 && catIndices.Length <= columnSlotsCount * 2)
                {
                    int previousEndIndex = -1;
                    isValid = true;
                    var catIndicesValues = catIndices.GetValues();
                    for (int i = 0; i < catIndicesValues.Length; i += 2)
                    {
                        if (catIndicesValues[i] > catIndicesValues[i + 1] ||
                            catIndicesValues[i] <= previousEndIndex ||
                            catIndicesValues[i] >= columnSlotsCount ||
                            catIndicesValues[i + 1] >= columnSlotsCount)
                        {
                            isValid = false;
                            break;
                        }

                        previousEndIndex = catIndicesValues[i + 1];
                    }
                    if (isValid)
                        categoricalFeatures = catIndicesValues.ToArray();
                }
            }

            return isValid;
        }

        /// <summary>
        /// Produces sequence of columns that are generated by trainer estimators.
        /// </summary>
        /// <param name="isNormalized">whether we should also append 'IsNormalized' (typically for probability column)</param>
        [BestFriend]
        internal static IEnumerable<SchemaShape.Column> GetTrainerOutputMetadata(bool isNormalized = false)
        {
            var cols = new List<SchemaShape.Column>();
            cols.Add(new SchemaShape.Column(Kinds.ScoreColumnSetId, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true));
            cols.Add(new SchemaShape.Column(Kinds.ScoreColumnKind, SchemaShape.Column.VectorKind.Scalar, TextType.Instance, false));
            cols.Add(new SchemaShape.Column(Kinds.ScoreValueKind, SchemaShape.Column.VectorKind.Scalar, TextType.Instance, false));
            if (isNormalized)
                cols.Add(new SchemaShape.Column(Kinds.IsNormalized, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false));
            return cols;
        }

        /// <summary>
        /// Produces metadata for the score column generated by trainer estimators for multiclass classification.
        /// If input LabelColumn is not available it produces slotnames metadata by default.
        /// </summary>
        /// <param name="labelColumn">Label column.</param>
        [BestFriend]
        internal static IEnumerable<SchemaShape.Column> MetadataForMulticlassScoreColumn(SchemaShape.Column? labelColumn = null)
        {
            var cols = new List<SchemaShape.Column>();
            if (labelColumn != null && labelColumn.Value.IsKey && HasKeyValues(labelColumn.Value))
                cols.Add(new SchemaShape.Column(Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextType.Instance, false));
            cols.AddRange(GetTrainerOutputMetadata());
            return cols;
        }

        private sealed class MetadataRow : Row
        {
            private readonly Schema.Metadata _metadata;

            public MetadataRow(Schema.Metadata metadata)
            {
                Contracts.AssertValue(metadata);
                _metadata = metadata;
            }

            public override Schema Schema => _metadata.Schema;
            public override long Position => 0;
            public override long Batch => 0;
            public override ValueGetter<TValue> GetGetter<TValue>(int col) => _metadata.GetGetter<TValue>(col);
            public override ValueGetter<RowId> GetIdGetter() => (ref RowId dst) => dst = default;
            public override bool IsColumnActive(int col) => true;
        }

        /// <summary>
        /// Presents a <see cref="Schema.Metadata"/> as a an <see cref="Row"/>.
        /// </summary>
        /// <param name="metadata">The metadata to wrap.</param>
        /// <returns>A row that wraps an input metadata.</returns>
        [BestFriend]
        internal static Row MetadataAsRow(Schema.Metadata metadata)
        {
            Contracts.CheckValue(metadata, nameof(metadata));
            return new MetadataRow(metadata);
        }
    }
}