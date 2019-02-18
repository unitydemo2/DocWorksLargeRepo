// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Legacy.Models
{
    public sealed partial class OnnxConverter
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Legacy.Models.OnnxConverter.Convert(Microsoft.ML.Legacy.PredictionModel)" -->
                        public void Convert(PredictionModel model)
        {
            var environment = new MLContext();
            environment.CheckValue(model, nameof(model));

            Experiment experiment = environment.CreateExperiment();
            experiment.Add(this);
            experiment.Compile();
            experiment.SetInput(Model, model.PredictorModel);
            experiment.Run();
        }
    }
}
