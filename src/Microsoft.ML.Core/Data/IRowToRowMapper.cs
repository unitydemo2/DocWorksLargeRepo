// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Data
{
    ///     <summary>
        ///     This interface maps an input <see cref="Row"/> to an output <see cref="Row"/>. Typically, the output contains
        ///     both the input columns and new columns added by the implementing class, although some implementations may
        ///     return a subset of the input columns.
        ///     This interface is similar to <see cref="ISchemaBoundRowMapper"/>, except it does not have any input role mappings,
        ///     so to rebind, the same input column names must be used.
        ///     Implementations of this interface are typically created over defined input <see cref="Schema"/>.
        ///     </summary>
            public interface IRowToRowMapper
    {
        ///     <summary>
                ///     Mappers are defined as accepting inputs with this very specific schema.
                ///     </summary>
                        Schema InputSchema { get; }

        ///     <summary>
                ///     Gets an instance of <see cref="Schema"/> which describes the columns' names and types in the output generated by this mapper.
                ///     </summary>
                        Schema OutputSchema { get; }

        ///     <summary>
                ///     Given a predicate specifying which columns are needed, return a predicate indicating which input columns are
                ///     needed. The domain of the function is defined over the indices of the columns of <see cref="Schema.Count"/>
                ///     for <see cref="InputSchema"/>.
                ///     </summary>
                        Func<int, bool> GetDependencies(Func<int, bool> predicate);

        ///      <summary>
                ///      Get an <see cref="Row"/> with the indicated active columns, based on the input <paramref name="input"/>.
                ///      The active columns are those for which <paramref name="active"/> returns true. Getting values on inactive
                ///      columns of the returned row will throw. Null predicates are disallowed.
                ///      The <see cref="Row.Schema"/> of <paramref name="input"/> should be the same object as
                ///      <see cref="InputSchema"/>. Implementors of this method should throw if that is not the case. Conversely,
                ///      the returned value must have the same schema as <see cref="OutputSchema"/>.
                ///      This method creates a live connection between the input <see cref="Row"/> and the output <see
                ///      cref="Row"/>. In particular, when the getters of the output <see cref="Row"/> are invoked, they invoke the
                ///      getters of the input row and base the output values on the current values of the input <see cref="Row"/>.
                ///      The output <see cref="Row"/> values are re-computed when requested through the getters. Also, the returned
                ///      <see cref="Row"/> will dispose <paramref name="input"/> when it is disposed.
                ///      </summary>
                        Row GetRow(Row input, Func<int, bool> active);
    }
}
