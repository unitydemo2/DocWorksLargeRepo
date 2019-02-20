// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Model;

namespace Microsoft.ML.Data
{
    ///     <summary>
        ///     Utilities to rebind data transforms
        ///     </summary>
            public static class ApplyTransformUtils
    {
        ///     <summary>
                ///     Attempt to apply the data transform to a different data view source.
                ///     If the transform in question implements <see cref="ITransformTemplate"/>, <see cref="ITransformTemplate.ApplyToData"/>
                ///     is called. Otherwise, the transform is serialized into a byte array and then deserialized.
                ///     </summary>
                ///     <param name="env">The host to use</param>
                ///     <param name="transform">The transform to apply.</param>
                ///     <param name="newSource">The data view to apply the transform to.</param>
                ///     <returns>The resulting data view.</returns>
                        public static IDataTransform ApplyTransformToData(IHostEnvironment env, IDataTransform transform, IDataView newSource)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transform, nameof(transform));
            env.CheckValue(newSource, nameof(newSource));
            var rebindable = transform as ITransformTemplate;
            if (rebindable != null)
                return rebindable.ApplyToData(env, newSource);

            // Revert to serialization.
            using (var stream = new MemoryStream())
            {
                using (var rep = RepositoryWriter.CreateNew(stream, env))
                {
                    ModelSaveContext.SaveModel(rep, transform, "model");
                    rep.Commit();
                }

                stream.Position = 0;
                using (var rep = RepositoryReader.Open(stream, env))
                {
                    IDataTransform newData;
                    ModelLoadContext.LoadModel<IDataTransform, SignatureLoadDataTransform>(env,
                        out newData, rep, "model", newSource);
                    return newData;
                }
            }
        }

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.Data.ApplyTransformUtils.ApplyAllTransformsToData(Microsoft.ML.IHostEnvironment,Microsoft.ML.Data.IDataView,Microsoft.ML.Data.IDataView,Microsoft.ML.Data.IDataView)" -->
                        public static IDataView ApplyAllTransformsToData(IHostEnvironment env, IDataView chain, IDataView newSource, IDataView oldSource = null)
        {
            // REVIEW: have a variation that would selectively apply transforms?
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(chain, nameof(chain));
            env.CheckValue(newSource, nameof(newSource));
            env.CheckValueOrNull(oldSource);

            // Backtrack the chain until we reach a chain start or a non-transform.
            // REVIEW: we 'unwrap' the composite data loader here and step through its pipeline.
            // It's probably more robust to make CompositeDataLoader not even be an IDataView, this
            // would force the user to do the right thing and unwrap on his end.
            var cdl = chain as CompositeDataLoader;
            if (cdl != null)
                chain = cdl.View;

            var transforms = new List<IDataTransform>();
            IDataTransform xf;
            while ((xf = chain as IDataTransform) != null)
            {
                if (chain == oldSource)
                    break;
                transforms.Add(xf);
                chain = xf.Source;

                cdl = chain as CompositeDataLoader;
                if (cdl != null)
                    chain = cdl.View;
            }
            transforms.Reverse();

            env.Check(oldSource == null || chain == oldSource, "Source data not found in the chain");

            IDataView newChain = newSource;
            foreach (var transform in transforms)
                newChain = ApplyTransformToData(env, transform, newChain);

            return newChain;
        }
    }
}
