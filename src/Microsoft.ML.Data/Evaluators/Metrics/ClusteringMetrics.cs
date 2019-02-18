// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    ///     <summary>
        ///     The metrics generated after evaluating the clustering predictions.
        ///     </summary>
            public sealed class ClusteringMetrics
    {
        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.Data.ClusteringMetrics.Nmi" -->
                        public double Nmi { get; }

        ///     <summary>
                ///     Average Score. For the K-Means algorithm, the 'score' is the distance from the centroid to the example.
                ///     The average score is, therefore, a measure of proximity of the examples to cluster centroids.
                ///     In other words, it's the 'cluster tightness' measure.
                ///     Note however, that this metric will only decrease if the number of clusters is increased,
                ///     and in the extreme case (where each distinct example is its own cluster) it will be equal to zero.
                ///     </summary>
                        public double AvgMinScore { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.Data.ClusteringMetrics.Dbi" -->
                        public double Dbi { get; }

        internal ClusteringMetrics(IExceptionContext ectx, Row overallResult, bool calculateDbi)
        {
            double Fetch(string name) => RowCursorUtils.Fetch<double>(ectx, overallResult, name);

            Nmi = Fetch(ClusteringEvaluator.Nmi);
            AvgMinScore = Fetch(ClusteringEvaluator.AvgMinScore);

            if (calculateDbi)
                Dbi = Fetch(ClusteringEvaluator.Dbi);
        }

        internal ClusteringMetrics(double nmi, double avgMinScore, double dbi)
        {
            Nmi = nmi;
            AvgMinScore = avgMinScore;
            Dbi = dbi;
        }
    }
}