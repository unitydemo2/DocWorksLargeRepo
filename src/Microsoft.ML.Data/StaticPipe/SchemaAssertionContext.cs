// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.StaticPipe.Runtime
{
    /// <!-- Badly formed XML comment ignored for member "T:Microsoft.ML.StaticPipe.Runtime.SchemaAssertionContext" -->
            public sealed class SchemaAssertionContext
    {
        // Hiding all these behind empty-structures is a bit of a cheap trick, but probably works
        // pretty well considering that the alternative is a bunch of tiny objects allocated on the
        // stack. Plus, the default value winds up working for them. We can also exploit the `ref struct`
        // property of these things to make sure people don't make the mistake of assigning them as the
        // values.

        ///     <summary>Assertions over a column of <see cref="NumberType.I1"/>.</summary>
                        public PrimitiveTypeAssertions<sbyte> I1 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.I2"/>.</summary>
                        public PrimitiveTypeAssertions<short> I2 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.I4"/>.</summary>
                        public PrimitiveTypeAssertions<int> I4 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.I8"/>.</summary>
                        public PrimitiveTypeAssertions<long> I8 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.U1"/>.</summary>
                        public PrimitiveTypeAssertions<byte> U1 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.U2"/>.</summary>
                        public PrimitiveTypeAssertions<ushort> U2 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.U4"/>.</summary>
                        public PrimitiveTypeAssertions<uint> U4 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.U8"/>.</summary>
                        public PrimitiveTypeAssertions<ulong> U8 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.R4"/>.</summary>
                        public NormalizableTypeAssertions<float> R4 => default;

        ///     <summary>Assertions over a column of <see cref="NumberType.R8"/>.</summary>
                        public NormalizableTypeAssertions<double> R8 => default;

        ///     <summary>Assertions over a column of <see cref="TextType"/>.</summary>
                        public PrimitiveTypeAssertions<string> Text => default;

        ///     <summary>Assertions over a column of <see cref="BoolType"/>.</summary>
                        public PrimitiveTypeAssertions<bool> Bool => default;

        ///     <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U1"/> <see cref="ColumnType.RawKind"/>.</summary>
                        public KeyTypeSelectorAssertions<byte> KeyU1 => default;
        ///     <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U2"/> <see cref="ColumnType.RawKind"/>.</summary>
                        public KeyTypeSelectorAssertions<ushort> KeyU2 => default;
        ///     <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U4"/> <see cref="ColumnType.RawKind"/>.</summary>
                        public KeyTypeSelectorAssertions<uint> KeyU4 => default;
        ///     <summary>Assertions over a column of <see cref="KeyType"/> with <see cref="DataKind.U8"/> <see cref="ColumnType.RawKind"/>.</summary>
                        public KeyTypeSelectorAssertions<ulong> KeyU8 => default;

        internal static SchemaAssertionContext Inst = new SchemaAssertionContext();

        private SchemaAssertionContext() { }

        // Until we have some transforms that use them, we might not expect to see too much interest in asserting
        // the time relevant datatypes.

        /// <summary>
        /// Holds assertions relating to the basic primitive types.
        /// </summary>
        public ref struct PrimitiveTypeAssertions<T>
        {
            private PrimitiveTypeAssertions(int i) { }

            /// <summary>
            /// Asserts a type that is directly this <see cref="PrimitiveType"/>.
            /// </summary>
            public Scalar<T> Scalar => null;

            /// <summary>
            /// Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
            /// where <see cref="ColumnType.IsKnownSizeVector"/> is true.
            /// </summary>
            public Vector<T> Vector => null;

            /// <summary>
            /// Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
            /// where <see cref="ColumnType.IsKnownSizeVector"/> is true.
            /// </summary>
            public VarVector<T> VarVector => null;
        }

        
        public ref struct NormalizableTypeAssertions<T>
        {
            private NormalizableTypeAssertions(int i) { }

            ///     <summary>
                        ///     Asserts a type that is directly this <see cref="PrimitiveType"/>.
                        ///     </summary>
                                    public Scalar<T> Scalar => null;

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
                        ///     where <see cref="ColumnType.IsKnownSizeVector"/> is true.
                        ///     </summary>
                                    public Vector<T> Vector => null;

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="PrimitiveType"/>,
                        ///     where <see cref="ColumnType.IsKnownSizeVector"/> is true.
                        ///     </summary>
                                    public VarVector<T> VarVector => null;
            /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.StaticPipe.Runtime.SchemaAssertionContext.NormalizableTypeAssertions`1.NormVector" -->
                                    public NormVector<T> NormVector => null;
        }

        ///     <summary>
                ///     Once a single general key type has been selected, we can select its vector-ness.
                ///     </summary>
                ///     <typeparam name="T">The static type corresponding to a <see cref="KeyType"/>.</typeparam>
                        public ref struct KeyTypeVectorAssertions<T>
            where T : class
        {
            private KeyTypeVectorAssertions(int i) { }

            ///     <summary>
                        ///     Asserts a type that is directly this <see cref="KeyType"/>.
                        ///     </summary>
                                    public T Scalar => null;

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="KeyType"/>,
                        ///     where <see cref="ColumnType.IsKnownSizeVector"/> is true.
                        ///     </summary>
                                    public Vector<T> Vector => null;

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="VectorType"/> of this <see cref="KeyType"/>,
                        ///     where <see cref="ColumnType.IsKnownSizeVector"/> is true.
                        ///     </summary>
                                    public VarVector<T> VarVector => null;
        }

        ///     <summary>
                ///     Assertions for key types of various forms. Used to select a particular <see cref="KeyTypeVectorAssertions{T}"/>.
                ///     </summary>
                ///     <typeparam name="T"></typeparam>
                        public ref struct KeyTypeSelectorAssertions<T>
        {
            private KeyTypeSelectorAssertions(int i) { }

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="KeyType"/> where <see cref="KeyType.Count"/> is positive, that is, is of known cardinality,
                        ///     but that we are not asserting has any particular type of <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.
                        ///     </summary>
                                    public KeyTypeVectorAssertions<Key<T>> NoValue => default;

            ///     <summary>
                        ///     Asserts a type corresponding to a <see cref="KeyType"/> where <see cref="KeyType.Count"/> is zero, that is, is of unknown cardinality.
                        ///     </summary>
                                    public KeyTypeVectorAssertions<VarKey<T>> UnknownCardinality => default;

            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I1"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, sbyte>> I1Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I2"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, short>> I2Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, int>> I4Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.I8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, long>> I8Values => default;

            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U1"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, byte>> U1Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U2"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, ushort>> U2Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, uint>> U4Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.U8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, ulong>> U8Values => default;

            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.R4"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, float>> R4Values => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="NumberType.R8"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, double>> R8Values => default;

            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="TextType"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, string>> TextValues => default;
            ///     <summary>Asserts a <see cref="KeyType"/> of known cardinality with a vector of <see cref="BoolType"/> <see cref="MetadataUtils.Kinds.KeyValues"/> metadata.</summary>
                                    public KeyTypeVectorAssertions<Key<T, bool>> BoolValues => default;
        }
    }
}