// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    
    public sealed class RankerMetrics
    {
        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.Data.RankerMetrics.Ndcg" -->
                        public double[] Ndcg { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.Data.RankerMetrics.Dcg" -->
                        public double[] Dcg { get; }

        private static T Fetch<T>(IExceptionContext ectx, Row row, string name)
        {
            if (!row.Schema.TryGetColumnIndex(name, out int col))
                throw ectx.Except($"Could not find column '{name}'");
            T val = default;
            row.GetGetter<T>(col)(ref val);
            return val;
        }

        internal RankerMetrics(IExceptionContext ectx, Row overallResult)
        {
            VBuffer<double> Fetch(string name) => Fetch<VBuffer<double>>(ectx, overallResult, name);

            Dcg = Fetch(RankerEvaluator.Dcg).GetValues().ToArray();
            Ndcg = Fetch(RankerEvaluator.Ndcg).GetValues().ToArray();
        }

        internal RankerMetrics(double[] dcg, double[] ndcg)
        {
            Dcg = new double[dcg.Length];
            dcg.CopyTo(Dcg, 0);
            Ndcg = new double[ndcg.Length];
            ndcg.CopyTo(Ndcg, 0);
        }
    }
}