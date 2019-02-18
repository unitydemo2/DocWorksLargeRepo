// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using static Microsoft.ML.EntryPoints.CommonInputs;

namespace Microsoft.ML.Legacy.Models
{
    public sealed partial class OneVersusAll
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Legacy.Models.OneVersusAll.With(Microsoft.ML.EntryPoints.CommonInputs.ITrainerInputWithLabel,System.Boolean)" -->
                        public static ILearningPipelineItem With(ITrainerInputWithLabel trainer, bool useProbabilities = true)
        {
            return new OvaPipelineItem(trainer, useProbabilities);
        }

        private class OvaPipelineItem : ILearningPipelineItem
        {
            private Var<IDataView> _data;
            private ITrainerInputWithLabel _trainer;
            private bool _useProbabilities;

            public OvaPipelineItem(ITrainerInputWithLabel trainer, bool useProbabilities)
            {
                _trainer = trainer;
                _useProbabilities = useProbabilities;
            }

            public ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment)
            {
                var env = new MLContext();
                var subgraph = env.CreateExperiment();
                subgraph.Add(_trainer);
                var ova = new OneVersusAll();
                if (previousStep != null)
                {
                    if (!(previousStep is ILearningPipelineDataStep dataStep))
                    {
                        throw new InvalidOperationException($"{ nameof(OneVersusAll)} only supports an { nameof(ILearningPipelineDataStep)} as an input.");
                    }

                    _data = dataStep.Data;
                    ova.TrainingData = dataStep.Data;
                    ova.UseProbabilities = _useProbabilities;
                    ova.Nodes = subgraph;
                }
                Output output = experiment.Add(ova);
                return new OvaPipelineStep(output);
            }

            public Var<IDataView> GetInputData() => _data;
        }

        private class OvaPipelineStep : ILearningPipelinePredictorStep
        {
            public OvaPipelineStep(Output output)
            {
                Model = output.PredictorModel;
            }

            public Var<PredictorModel> Model { get; }
        }
    }
}
