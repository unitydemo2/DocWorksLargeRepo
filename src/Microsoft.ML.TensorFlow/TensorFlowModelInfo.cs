// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TensorFlow;

namespace Microsoft.ML.Transforms
{
    /// <!-- Badly formed XML comment ignored for member "T:Microsoft.ML.Transforms.TensorFlowModelInfo" -->
            public class TensorFlowModelInfo
    {
        internal TFSession Session { get; }
        
        public string ModelPath { get; }

        private readonly IHostEnvironment _env;

        /// <summary>
        /// Instantiates <see cref="TensorFlowModelInfo"/>.
        /// </summary>
        /// <param name="env">An <see cref="IHostEnvironment"/> object.</param>
        /// <param name="session">TensorFlow session object.</param>
        /// <param name="modelLocation">Location of the model from where <paramref name="session"/> was loaded.</param>
        internal TensorFlowModelInfo(IHostEnvironment env, TFSession session, string modelLocation)
        {
            Session = session;
            ModelPath = modelLocation;
            _env = env;
        }

        /// <summary>
        /// Get <see cref="ISchema"/> for complete model. Every node in the TensorFlow model will be included in the <see cref="ISchema"/> object.
        /// </summary>
        internal Schema GetModelSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.Graph);
        }

        ///     <summary>
                ///     Get <see cref="ISchema"/> for only those nodes which are marked "Placeholder" in the TensorFlow model.
                ///     This method is convenient for exploring the model input(s) in case TensorFlow graph is very large.
                ///     </summary>
                        public Schema GetInputSchema()
        {
            return TensorFlowUtils.GetModelSchema(_env, Session.Graph, "Placeholder");
        }
    }
}
