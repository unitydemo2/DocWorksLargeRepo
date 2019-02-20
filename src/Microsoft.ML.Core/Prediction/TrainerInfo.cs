// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML
{
    ///     <summary>
        ///     Instances of this class posses information about trainers, in terms of their requirements and capabilities.
        ///     The intended usage is as the value for <see cref="ITrainer.Info"/>.
        ///     </summary>
            public sealed class TrainerInfo
    {
        // REVIEW: Ideally trainers should be able to communicate
        // something about the type of data they are capable of being trained
        // on, for example, what ColumnKinds they want, how many of each, of what type,
        // etc. This interface seems like the most natural conduit for that sort
        // of extra information.

        ///     <summary>
                ///     Whether the trainer needs to see data in normalized form. Only non-parametric learners will tend to produce
                ///     normalization here.
                ///     </summary>
                        public bool NeedNormalization { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.TrainerInfo.NeedCalibration" -->
                        public bool NeedCalibration { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.TrainerInfo.WantCaching" -->
                        public bool WantCaching { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.TrainerInfo.SupportsValidation" -->
                        public bool SupportsValidation { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.TrainerInfo.SupportsTest" -->
                        public bool SupportsTest { get; }

        /// <!-- Badly formed XML comment ignored for member "P:Microsoft.ML.TrainerInfo.SupportsIncrementalTraining" -->
                        public bool SupportsIncrementalTraining { get; }

        ///     <summary>
                ///     Initializes with the given parameters. The parameters have default values for the most typical values
                ///     for most classical trainers.
                ///     </summary>
                ///     <param name="normalization">The value for the property <see cref="NeedNormalization"/></param>
                ///     <param name="calibration">The value for the property <see cref="NeedCalibration"/></param>
                ///     <param name="caching">The value for the property <see cref="WantCaching"/></param>
                ///     <param name="supportValid">The value for the property <see cref="SupportsValidation"/></param>
                ///     <param name="supportIncrementalTrain">The value for the property <see cref="SupportsIncrementalTraining"/></param>
                ///     <param name="supportTest">The value for the property <see cref="SupportsTest"/></param>
                        public TrainerInfo(bool normalization = true, bool calibration = false, bool caching = true,
            bool supportValid = false, bool supportIncrementalTrain = false, bool supportTest = false)
        {
            NeedNormalization = normalization;
            NeedCalibration = calibration;
            WantCaching = caching;
            SupportsValidation = supportValid;
            SupportsIncrementalTraining = supportIncrementalTrain;
            SupportsTest = supportTest;
        }
    }
}
