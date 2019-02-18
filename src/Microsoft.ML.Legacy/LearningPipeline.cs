// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

namespace Microsoft.ML.Legacy
{
    [Obsolete]
    public sealed class ScorerPipelineStep : ILearningPipelineDataStep
    {
        public ScorerPipelineStep(Var<IDataView> data, Var<TransformModel> model)
        {
            Data = data;
            Model = model;
        }

        public Var<IDataView> Data { get; }
        public Var<TransformModel> Model { get; }
    }

    /// <!-- Badly formed XML comment ignored for member "T:Microsoft.ML.Legacy.LearningPipeline" -->
            [Obsolete]
    [DebuggerTypeProxy(typeof(LearningPipelineDebugProxy))]
    public class LearningPipeline : ICollection<ILearningPipelineItem>
    {
        private List<ILearningPipelineItem> Items { get; }
        private readonly int? _seed;
        private readonly int _conc;

        ///     <summary>
                ///     Construct an empty <see cref="LearningPipeline"/> object.
                ///     </summary>
                        public LearningPipeline()
            : this(conc: 0)
        {
        }

        /// <summary>
        ///  Construct an empty <see cref="LearningPipeline"/> object.
        /// </summary>
        /// <param name="seed">Specify seed for random generator</param>
        /// <param name="conc">Specify concurrency factor (default value - autoselection)</param>
        internal LearningPipeline(int? seed = null, int conc = 0)
        {
            Items = new List<ILearningPipelineItem>();
            _seed = seed;
            _conc = conc;
        }

        ///     <summary>
                ///     Get the count of ML components in the <see cref="LearningPipeline"/> object
                ///     </summary>
                        public int Count => Items.Count;
        
        public bool IsReadOnly => false;

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Legacy.LearningPipeline.Add(Microsoft.ML.Legacy.ILearningPipelineItem)" -->
                        public void Add(ILearningPipelineItem item) => Items.Add(item);

        ///     <summary>
                ///     Add a data loader, transform or trainer into the pipeline.
                ///     </summary>
                ///     <param name="item">Any ML component (data loader, transform or trainer) defined as <see cref="ILearningPipelineItem"/>.</param>
                ///     <returns>Pipeline with added item</returns>
                        public LearningPipeline Append(ILearningPipelineItem item)
        {
            Add(item);
            return this;
        }
        ///     <summary>
                ///     Remove all the loaders/transforms/trainers from the pipeline.
                ///     </summary>
                        public void Clear() => Items.Clear();

        ///     <summary>
                ///     Check if a specific loader/transform/trainer is in the pipeline?
                ///     </summary>
                ///     <param name="item">Any ML component (data loader, transform or trainer) defined as <see cref="ILearningPipelineItem"/>.</param>
                ///     <returns>true if item is found in the pipeline; otherwise, false.</returns>
                        public bool Contains(ILearningPipelineItem item) => Items.Contains(item);

        ///     <summary>
                ///     Copy the pipeline items into an array.
                ///     </summary>
                ///     <param name="array">The one-dimensional Array that is the destination of the elements copied from.</param>
                ///     <param name="arrayIndex">The zero-based index in <paramref name="array" /> at which copying begins.</param>
                        public void CopyTo(ILearningPipelineItem[] array, int arrayIndex) => Items.CopyTo(array, arrayIndex);
        
        public IEnumerator<ILearningPipelineItem> GetEnumerator() => Items.GetEnumerator();

        ///     <summary>
                ///     Remove an item from the pipeline.
                ///     </summary>
                ///     <param name="item"><see cref="ILearningPipelineItem"/> to remove.</param>
                ///     <returns>true if item was removed from the pipeline; otherwise, false.</returns>
                        public bool Remove(ILearningPipelineItem item) => Items.Remove(item);
        
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        ///     <summary>
                ///     Train the model using the ML components in the pipeline.
                ///     </summary>
                ///     <typeparam name="TInput">Type of data instances the model will be trained on. It's a custom type defined by the user according to the structure of data.
                ///     <para/>
                ///     Please see https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet/get-started/windows for more details on input type.
                ///     </typeparam>
                ///     <typeparam name="TOutput">Ouput type. The prediction will be return based on this type.
                ///     Please see https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet/get-started/windows for more details on output type.
                ///     </typeparam>
                ///     <returns>PredictionModel object. This is the model object used for prediction on new instances. </returns>
                        public PredictionModel<TInput, TOutput> Train<TInput, TOutput>()
            where TInput : class
            where TOutput : class, new()
        {
            var environment = new MLContext(seed: _seed, conc: _conc);
            Experiment experiment = environment.CreateExperiment();
            ILearningPipelineStep step = null;
            List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
            List<Var<TransformModel>> transformModels = new List<Var<TransformModel>>();
            Var<TransformModel> lastTransformModel = null;

            foreach (ILearningPipelineItem currentItem in this)
            {
                if (currentItem is ILearningPipelineLoader loader)
                    loaders.Add(loader);

                step = currentItem.ApplyStep(step, experiment);
                if (step is ILearningPipelineDataStep dataStep && dataStep.Model != null)
                    transformModels.Add(dataStep.Model);
                else if (step is ILearningPipelinePredictorStep predictorDataStep)
                {
                    if (lastTransformModel != null)
                        transformModels.Insert(0, lastTransformModel);

                    Var<PredictorModel> predictorModel;
                    if (transformModels.Count != 0)
                    {
                        var localModelInput = new Transforms.ManyHeterogeneousModelCombiner
                        {
                            PredictorModel = predictorDataStep.Model,
                            TransformModels = new ArrayVar<TransformModel>(transformModels.ToArray())
                        };
                        var localModelOutput = experiment.Add(localModelInput);
                        predictorModel = localModelOutput.PredictorModel;
                    }
                    else
                        predictorModel = predictorDataStep.Model;

                    var scorer = new Transforms.Scorer
                    {
                        PredictorModel = predictorModel
                    };

                    var scorerOutput = experiment.Add(scorer);
                    lastTransformModel = scorerOutput.ScoringTransform;
                    step = new ScorerPipelineStep(scorerOutput.ScoredData, scorerOutput.ScoringTransform);
                    transformModels.Clear();
                }
            }

            if (transformModels.Count > 0)
            {
                if (lastTransformModel != null)
                    transformModels.Insert(0, lastTransformModel);

                var modelInput = new Transforms.ModelCombiner
                {
                    Models = new ArrayVar<TransformModel>(transformModels.ToArray())
                };

                var modelOutput = experiment.Add(modelInput);
                lastTransformModel = modelOutput.OutputModel;
            }

            experiment.Compile();
            foreach (ILearningPipelineLoader loader in loaders)
            {
                loader.SetInput(environment, experiment);
            }
            experiment.Run();

            TransformModel model = experiment.GetOutput(lastTransformModel);
            BatchPredictionEngine<TInput, TOutput> predictor;
            using (var memoryStream = new MemoryStream())
            {
                model.Save(environment, memoryStream);

                memoryStream.Position = 0;

                predictor = environment.CreateBatchPredictionEngine<TInput, TOutput>(memoryStream);

                return new PredictionModel<TInput, TOutput>(predictor, memoryStream);
            }
        }

        /// <summary>
        /// Executes a pipeline and returns the resulting data.
        /// </summary>
        /// <returns>
        /// The IDataView that was returned by the pipeline.
        /// </returns>
        internal IDataView Execute(IHostEnvironment environment)
        {
            Experiment experiment = environment.CreateExperiment();
            ILearningPipelineStep step = null;
            List<ILearningPipelineLoader> loaders = new List<ILearningPipelineLoader>();
            foreach (ILearningPipelineItem currentItem in this)
            {
                if (currentItem is ILearningPipelineLoader loader)
                    loaders.Add(loader);

                step = currentItem.ApplyStep(step, experiment);
            }

            if (!(step is ILearningPipelineDataStep endDataStep))
            {
                throw new InvalidOperationException($"{nameof(LearningPipeline)}.{nameof(Execute)} must have a Data step as the last step.");
            }

            experiment.Compile();
            foreach (ILearningPipelineLoader loader in loaders)
            {
                loader.SetInput(environment, experiment);
            }
            experiment.Run();

            return experiment.GetOutput(endDataStep.Data);
        }
    }
}
