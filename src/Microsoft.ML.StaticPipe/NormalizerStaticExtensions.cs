// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Normalizers;

namespace Microsoft.ML.StaticPipe
{
    ///     <summary>
    ///     Extension methods for static pipelines for normalization of data.
    ///     </summary>
        public static class NormalizerStaticExtensions
    {
        private const long MaxTrain = NormalizingEstimator.Defaults.MaxTrainingExamples;
        private const bool FZ = NormalizingEstimator.Defaults.FixZero;

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.Normalize(Microsoft.ML.StaticPipe.Vector{System.Single},System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitAffine{System.Collections.Immutable.ImmutableArray{System.Single}})" -->
                        public static NormVector<float> Normalize(
            this Vector<float> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.Normalize(Microsoft.ML.StaticPipe.Vector{System.Double},System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitAffine{System.Collections.Immutable.ImmutableArray{System.Double}})" -->
                        public static NormVector<double> Normalize(
            this Vector<double> input, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMinMaxCore(input, fixZero, maxTrainingExamples, onFit);
        }

        private static NormVector<T> NormalizeByMinMaxCore<T>(Vector<T> input, bool fixZero, long maxTrainingExamples,
            OnFitAffine<ImmutableArray<T>> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) => new NormalizingEstimator.MinMaxColumn(src, name, maxTrainingExamples, fixZero), AffineMapper(onFit));
        }

        // We have a slightly different breaking up of categories of normalizers versus the dynamic API. Both the mean-var and
        // CDF normalizers are initialized in the same way because they gather exactly the same statistics, but from the point of
        // view of the static API what is more important is the type of mapping that winds up being computed.

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByMeanVar(Microsoft.ML.StaticPipe.Vector{System.Single},System.Boolean,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitAffine{System.Collections.Immutable.ImmutableArray{System.Single}})" -->
                        public static NormVector<float> NormalizeByMeanVar(
            this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByMeanVar(Microsoft.ML.StaticPipe.Vector{System.Double},System.Boolean,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitAffine{System.Collections.Immutable.ImmutableArray{System.Double}})" -->
                        public static NormVector<double> NormalizeByMeanVar(
            this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitAffine<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, false, maxTrainingExamples, AffineMapper(onFit));
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByCumulativeDistribution(Microsoft.ML.StaticPipe.Vector{System.Single},System.Boolean,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitCumulativeDistribution{System.Collections.Immutable.ImmutableArray{System.Single}})" -->
                        public static NormVector<float> NormalizeByCumulativeDistribution(
            this Vector<float> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitCumulativeDistribution<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByCumulativeDistribution(Microsoft.ML.StaticPipe.Vector{System.Double},System.Boolean,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitCumulativeDistribution{System.Collections.Immutable.ImmutableArray{System.Double}})" -->
                        public static NormVector<double> NormalizeByCumulativeDistribution(
            this Vector<double> input, bool fixZero = FZ, bool useLog = false, long maxTrainingExamples = MaxTrain,
            OnFitCumulativeDistribution<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByMVCdfCore(input, fixZero, useLog, true, maxTrainingExamples, CdfMapper(onFit));
        }

        private static NormVector<T> NormalizeByMVCdfCore<T>(Vector<T> input, bool fixZero, bool useLog, bool useCdf, long maxTrainingExamples, Action<IColumnFunction> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) =>
            {
                if (useLog)
                    return new NormalizingEstimator.LogMeanVarColumn(src, name, maxTrainingExamples, useCdf);
                return new NormalizingEstimator.MeanVarColumn(src, name, maxTrainingExamples, fixZero, useCdf);
            }, onFit);
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByBinning(Microsoft.ML.StaticPipe.Vector{System.Single},System.Int32,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitBinned{System.Collections.Immutable.ImmutableArray{System.Single}})" -->
                        public static NormVector<float> NormalizeByBinning(
            this Vector<float> input, int maxBins = NormalizingEstimator.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitBinned<ImmutableArray<float>> onFit = null)
        {
            return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.NormalizeByBinning(Microsoft.ML.StaticPipe.Vector{System.Double},System.Int32,System.Boolean,System.Int64,Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitBinned{System.Collections.Immutable.ImmutableArray{System.Double}})" -->
                        public static NormVector<double> NormalizeByBinning(
            this Vector<double> input, int maxBins = NormalizingEstimator.Defaults.NumBins, bool fixZero = FZ, long maxTrainingExamples = MaxTrain,
            OnFitBinned<ImmutableArray<double>> onFit = null)
        {
            return NormalizeByBinningCore(input, maxBins, fixZero, maxTrainingExamples, onFit);
        }

        private static NormVector<T> NormalizeByBinningCore<T>(Vector<T> input, int numBins, bool fixZero, long maxTrainingExamples,
            OnFitBinned<ImmutableArray<T>> onFit)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckParam(numBins > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            Contracts.CheckParam(maxTrainingExamples > 1, nameof(maxTrainingExamples), "Must be greater than 1");
            return new Impl<T>(input, (src, name) => new NormalizingEstimator.BinningColumn(src, name, maxTrainingExamples, fixZero, numBins), BinMapper(onFit));
        }

        /// <!-- Badly formed XML comment ignored for member "T:Microsoft.ML.StaticPipe.NormalizerStaticExtensions.OnFitAffine`1" -->
                        public delegate void OnFitAffine<TData>(TData scale, TData offset);

        /// <summary>
        /// For user provided delegates to receive information when a cumulative distribution function normalizer is fitted.
        /// </summary>
        /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
        /// <param name="mean">The mean value. In the scalar case, this is a single value. In the vector case this is of length equal
        /// to the number of slots.</param>
        /// <param name="standardDeviation">The standard deviation. In the scalar case, this is a single value. In the vector case
        /// this is of length equal to the number of slots.</param>
        public delegate void OnFitCumulativeDistribution<TData>(TData mean, TData standardDeviation);

        /// <summary>
        /// For user provided delegates to receive information when a binning normalizer is fitted.
        /// The function fo the normalizer transformer is, given a value, find its index in the upper bounds, then divide that value
        /// by the number of upper bounds minus 1, so as to scale the index between 0 and 1. Then, if zero had been fixed, subtract
        /// off the value that would have been computed by the above procedure for the value zero.
        /// </summary>
        /// <typeparam name="TData">The data type being received, either a numeric type, or a sequence of the numeric type</typeparam>
        /// <param name="upperBounds">For a scalar column a single sequence of the bin upper bounds. For a vector, the same, but
        /// for all slots.</param>
        public delegate void OnFitBinned<TData>(ImmutableArray<TData> upperBounds);

        #region Implementation support
        private delegate NormalizingEstimator.ColumnBase CreateNormCol(string input, string name);

        private sealed class Rec : EstimatorReconciler
        {
            // All settings are self contained in the columns.
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var cols = new NormalizingEstimator.ColumnBase[toOutput.Length];
                List<(int idx, Action<IColumnFunction> onFit)> onFits = null;

                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (INormColCreator)toOutput[i];
                    cols[i] = col.CreateNormCol(inputNames[col.Input], outputNames[toOutput[i]]);
                    if (col.OnFit != null)
                        Utils.Add(ref onFits, (i, col.OnFit));
                }
                var norm = new NormalizingEstimator(env, cols);
                if (Utils.Size(onFits) == 0)
                    return norm;
                return norm.WithOnFitDelegate(normTrans =>
                {
                    Contracts.Assert(normTrans.ColumnFunctions.Count == toOutput.Length);
                    foreach ((int idx, Action<IColumnFunction> onFit) in onFits)
                        onFit(normTrans.ColumnFunctions[idx]);
                });
            }
        }

        private static Action<IColumnFunction> AffineMapper<TData>(OnFitAffine<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizingTransformer.AffineNormalizerModelParameters<TData>)col?.GetNormalizerModelParams();
                onFit(aCol.Scale, aCol.Offset);
            };
        }

        private static Action<IColumnFunction> CdfMapper<TData>(OnFitCumulativeDistribution<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizingTransformer.CdfNormalizerModelParameters<TData>)col?.GetNormalizerModelParams();
                onFit(aCol.Mean, aCol.Stddev);
            };
        }

        private static Action<IColumnFunction> BinMapper<TData>(OnFitBinned<TData> onFit)
        {
            Contracts.AssertValueOrNull(onFit);
            if (onFit == null)
                return null;
            return col =>
            {
                var aCol = (NormalizingTransformer.BinNormalizerModelParameters<TData>)col?.GetNormalizerModelParams();
                onFit(aCol.UpperBounds);
            };
        }

        private interface INormColCreator
        {
            CreateNormCol CreateNormCol { get; }
            PipelineColumn Input { get; }
            Action<IColumnFunction> OnFit { get; }
        }

        private sealed class Impl<T> : NormVector<T>, INormColCreator
        {
            public PipelineColumn Input { get; }
            public CreateNormCol CreateNormCol { get; }
            public Action<IColumnFunction> OnFit { get; }

            public Impl(Vector<T> input, CreateNormCol del, Action<IColumnFunction> onFitDel)
                : base(Rec.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(del);
                Contracts.AssertValueOrNull(onFitDel);
                Input = input;
                CreateNormCol = del;
                OnFit = onFitDel;
            }
        }
        #endregion
    }
}
