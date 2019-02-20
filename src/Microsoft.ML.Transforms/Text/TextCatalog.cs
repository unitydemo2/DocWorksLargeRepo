// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Microsoft.ML
{
    using CharTokenizingDefaults = TokenizingByCharactersEstimator.Defaults;
    using TextNormalizeDefaults = TextNormalizingEstimator.Defaults;

    
    public static class TextCatalog
    {
        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.FeaturizeText(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.Action{Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Settings})" -->
                        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            Action<TextFeaturizingEstimator.Settings> advancedSettings = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, advancedSettings);

        ///     <summary>
                ///     Transform several text columns into featurized float array that represents counts of ngrams and char-grams.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumns">The input columns</param>
                ///     <param name="outputColumn">The output column</param>
                ///     <param name="advancedSettings">Advanced transform settings</param>
                        public static TextFeaturizingEstimator FeaturizeText(this TransformsCatalog.TextTransforms catalog,
            IEnumerable<string> inputColumns,
            string outputColumn,
            Action<TextFeaturizingEstimator.Settings> advancedSettings = null)
            => new TextFeaturizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumns, outputColumn, advancedSettings);

        ///     <summary>
                ///     Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumn">The column containing text to tokenize.</param>
                ///     <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
                ///     <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
                        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                useMarkerCharacters, new[] { (inputColumn, outputColumn) });

        ///     <summary>
        ///     Tokenize incoming text in input columns and output the tokens as output columns.
        ///     </summary>
        ///     <param name="catalog">The text-related transform's catalog.</param>
        ///     <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        ///     <param name="columns">Pairs of columns to run the tokenization on.</param>
        
        public static TokenizingByCharactersEstimator TokenizeCharacters(this TransformsCatalog.TextTransforms catalog,
            bool useMarkerCharacters = CharTokenizingDefaults.UseMarkerCharacters,
            params (string inputColumn, string outputColumn)[] columns)
            => new TokenizingByCharactersEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), useMarkerCharacters, columns);

        ///     <summary>
                ///     Normalizes incoming text in <paramref name="inputColumn"/> by changing case, removing diacritical marks, punctuation marks and/or numbers
                ///     and outputs new text as <paramref name="outputColumn"/>.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumn">The column containing text to normalize.</param>
                ///     <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
                ///     <param name="textCase">Casing text using the rules of the invariant culture.</param>
                ///     <param name="keepDiacritics">Whether to keep diacritical marks or remove them.</param>
                ///     <param name="keepPunctuations">Whether to keep punctuation marks or remove them.</param>
                ///     <param name="keepNumbers">Whether to keep numbers or remove them.</param>
                        public static TextNormalizingEstimator NormalizeText(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            TextNormalizingEstimator.CaseNormalizationMode textCase = TextNormalizeDefaults.TextCase,
            bool keepDiacritics = TextNormalizeDefaults.KeepDiacritics,
            bool keepPunctuations = TextNormalizeDefaults.KeepPunctuations,
            bool keepNumbers = TextNormalizeDefaults.KeepNumbers)
            => new TextNormalizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, textCase, keepDiacritics, keepPunctuations, keepNumbers);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ExtractWordEmbeddings(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,Microsoft.ML.Transforms.Text.WordEmbeddingsExtractingTransformer.PretrainedModelKind)" -->
                        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            WordEmbeddingsExtractingTransformer.PretrainedModelKind modelKind = WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, modelKind);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ExtractWordEmbeddings(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.String)" -->
                        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string customModelFile,
            string outputColumn = null)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, customModelFile);

        ///     <summary>
                ///     Extracts word embeddings.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="modelKind">The embeddings <see cref="WordEmbeddingsExtractingTransformer.PretrainedModelKind"/> to use. </param>
                ///     <param name="columns">The array columns, and per-column configurations to extract embeedings from.</param>
                        public static WordEmbeddingsExtractingEstimator ExtractWordEmbeddings(this TransformsCatalog.TextTransforms catalog,
           WordEmbeddingsExtractingTransformer.PretrainedModelKind modelKind = WordEmbeddingsExtractingTransformer.PretrainedModelKind.Sswe,
           params WordEmbeddingsExtractingTransformer.ColumnInfo[] columns)
            => new WordEmbeddingsExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), modelKind, columns);

        ///     <summary>
                ///     Tokenizes incoming text in <paramref name="inputColumn"/>, using <paramref name="separators"/> as separators,
                ///     and outputs the tokens as <paramref name="outputColumn"/>.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumn">The column containing text to tokenize.</param>
                ///     <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
                ///     <param name="separators">The separators to use (uses space character by default).</param>
                        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, separators);

        ///     <summary>
                ///     Tokenizes incoming text in input columns and outputs the tokens using <paramref name="separators"/> as separators.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to run the tokenization on.</param>
                ///     <param name="separators">The separators to use (uses space character by default).</param>
                        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            (string inputColumn, string outputColumn)[] columns,
            char[] separators = null)
            => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, separators);

        ///     <summary>
                ///      Tokenizes incoming text in input columns, using per-column configurations, and outputs the tokens.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to run the tokenization on.</param>
                        public static WordTokenizingEstimator TokenizeWords(this TransformsCatalog.TextTransforms catalog,
            params WordTokenizingTransformer.ColumnInfo[] columns)
          => new WordTokenizingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceNgrams(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.Int32,System.Int32,System.Boolean,System.Int32,Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria)" -->
                        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.Defaults.Weighting) =>
            new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn,
                ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        ///     <summary>
                ///     Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
                ///     and outputs bag of word vector for each output in <paramref name="columns.output"/>
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to compute bag of word vector.</param>
                ///     <param name="ngramLength">Ngram length.</param>
                ///     <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
                ///     <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
                ///     <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
                ///     <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
                        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
            (string input, string output)[] columns,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.Defaults.Weighting)
            => new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns,
                ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        ///     <summary>
                ///     Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
                ///     and outputs bag of word vector for each output in <paramref name="columns.output"/>
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to run the ngram process on.</param>
                        public static NgramExtractingEstimator ProduceNgrams(this TransformsCatalog.TextTransforms catalog,
             params NgramExtractingTransformer.ColumnInfo[] columns)
          => new NgramExtractingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns);

        ///     <summary>
                ///     Removes stop words from incoming token streams in <paramref name="inputColumn"/>
                ///     and outputs the token streams without stopwords as <paramref name="outputColumn"/>.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumn">The column containing text to remove stop words on.</param>
                ///     <param name="outputColumn">The column containing output text. Null means <paramref name="inputColumn"/> is replaced.</param>
                ///     <param name="language">Langauge of the input text column <paramref name="inputColumn"/>.</param>
                        public static StopWordsRemovingEstimator RemoveStopWords(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English)
            => new StopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), inputColumn, outputColumn, language);

        ///     <summary>
                ///     Removes stop words from incoming token streams in input columns
                ///     and outputs the token streams without stop words as output columns.
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to remove stop words on.</param>
                ///     <param name="language">Langauge of the input text columns <paramref name="columns"/>.</param>
                        public static StopWordsRemovingEstimator RemoveStopWords(this TransformsCatalog.TextTransforms catalog,
            (string input, string output)[] columns,
             StopWordsRemovingEstimator.Language language = StopWordsRemovingEstimator.Language.English)
            => new StopWordsRemovingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, language);

        ///     <summary>
                ///     Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumn"/>
                ///     and outputs bag of word vector as <paramref name="outputColumn"/>
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumn">The column containing text to compute bag of word vector.</param>
                ///     <param name="outputColumn">The column containing bag of word vector. Null means <paramref name="inputColumn"/> is replaced.</param>
                ///     <param name="ngramLength">Ngram length.</param>
                ///     <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
                ///     <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
                ///     <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
                ///     <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
                        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, ngramLength, skipLength, allLengths, maxNumTerms);

        ///     <summary>
                ///     Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="inputColumns"/>
                ///     and outputs bag of word vector as <paramref name="outputColumn"/>
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="inputColumns">The columns containing text to compute bag of word vector.</param>
                ///     <param name="outputColumn">The column containing output tokens.</param>
                ///     <param name="ngramLength">Ngram length.</param>
                ///     <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
                ///     <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
                ///     <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
                ///     <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
                        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            string[] inputColumns,
            string outputColumn,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumns, outputColumn, ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        ///     <summary>
                ///     Produces a bag of counts of ngrams (sequences of consecutive words) in <paramref name="columns.inputs"/>
                ///     and outputs bag of word vector for each output in <paramref name="columns.output"/>
                ///     </summary>
                ///     <param name="catalog">The text-related transform's catalog.</param>
                ///     <param name="columns">Pairs of columns to compute bag of word vector.</param>
                ///     <param name="ngramLength">Ngram length.</param>
                ///     <param name="skipLength">Maximum number of tokens to skip when constructing an ngram.</param>
                ///     <param name="allLengths">Whether to include all ngram lengths up to <paramref name="ngramLength"/> or only <paramref name="ngramLength"/>.</param>
                ///     <param name="maxNumTerms">Maximum number of ngrams to store in the dictionary.</param>
                ///     <param name="weighting">Statistical measure used to evaluate how important a word is to a document in a corpus.</param>
                        public static WordBagEstimator ProduceWordBags(this TransformsCatalog.TextTransforms catalog,
            (string[] inputs, string output)[] columns,
            int ngramLength = NgramExtractingEstimator.Defaults.NgramLength,
            int skipLength = NgramExtractingEstimator.Defaults.SkipLength,
            bool allLengths = NgramExtractingEstimator.Defaults.AllLengths,
            int maxNumTerms = NgramExtractingEstimator.Defaults.MaxNumTerms,
            NgramExtractingEstimator.WeightingCriteria weighting = NgramExtractingEstimator.WeightingCriteria.Tf)
            => new WordBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(), columns, ngramLength, skipLength, allLengths, maxNumTerms, weighting);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedWordBags(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedWordBags(Microsoft.ML.TransformsCatalog.TextTransforms,System.String[],System.String,System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            string[] inputColumns,
            string outputColumn,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumns, outputColumn, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedWordBags(Microsoft.ML.TransformsCatalog.TextTransforms,System.ValueTuple{System.String[],System.String}[],System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static WordHashBagEstimator ProduceHashedWordBags(this TransformsCatalog.TextTransforms catalog,
            (string[] inputs, string output)[] columns,
            int hashBits = NgramHashExtractingTransformer.DefaultArguments.HashBits,
            int ngramLength = NgramHashExtractingTransformer.DefaultArguments.NgramLength,
            int skipLength = NgramHashExtractingTransformer.DefaultArguments.SkipLength,
            bool allLengths = NgramHashExtractingTransformer.DefaultArguments.AllLengths,
            uint seed = NgramHashExtractingTransformer.DefaultArguments.Seed,
            bool ordered = NgramHashExtractingTransformer.DefaultArguments.Ordered,
            int invertHash = NgramHashExtractingTransformer.DefaultArguments.InvertHash)
            => new WordHashBagEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
               columns, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedNgrams(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
            => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                inputColumn, outputColumn, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedNgrams(Microsoft.ML.TransformsCatalog.TextTransforms,System.String[],System.String,System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            string[] inputColumns,
            string outputColumn,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
             => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                 inputColumns, outputColumn, hashBits, ngramLength, skipLength, allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.ProduceHashedNgrams(Microsoft.ML.TransformsCatalog.TextTransforms,System.ValueTuple{System.String[],System.String}[],System.Int32,System.Int32,System.Int32,System.Boolean,System.UInt32,System.Boolean,System.Int32)" -->
                        public static NgramHashingEstimator ProduceHashedNgrams(this TransformsCatalog.TextTransforms catalog,
            (string[] inputs, string output)[] columns,
            int hashBits = NgramHashingEstimator.Defaults.HashBits,
            int ngramLength = NgramHashingEstimator.Defaults.NgramLength,
            int skipLength = NgramHashingEstimator.Defaults.SkipLength,
            bool allLengths = NgramHashingEstimator.Defaults.AllLengths,
            uint seed = NgramHashingEstimator.Defaults.Seed,
            bool ordered = NgramHashingEstimator.Defaults.Ordered,
            int invertHash = NgramHashingEstimator.Defaults.InvertHash)
             => new NgramHashingEstimator(Contracts.CheckRef(catalog, nameof(catalog)).GetEnvironment(),
                 columns, hashBits, ngramLength, skipLength,allLengths, seed, ordered, invertHash);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.LatentDirichletAllocation(Microsoft.ML.TransformsCatalog.TextTransforms,System.String,System.String,System.Int32,System.Single,System.Single,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Boolean)" -->
                        public static LatentDirichletAllocationEstimator LatentDirichletAllocation(this TransformsCatalog.TextTransforms catalog,
            string inputColumn,
            string outputColumn = null,
            int numTopic = LatentDirichletAllocationEstimator.Defaults.NumTopic,
            float alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
            float beta = LatentDirichletAllocationEstimator.Defaults.Beta,
            int mhstep = LatentDirichletAllocationEstimator.Defaults.Mhstep,
            int numIterations = LatentDirichletAllocationEstimator.Defaults.NumIterations,
            int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
            int numThreads = LatentDirichletAllocationEstimator.Defaults.NumThreads,
            int numMaxDocToken = LatentDirichletAllocationEstimator.Defaults.NumMaxDocToken,
            int numSummaryTermPerTopic = LatentDirichletAllocationEstimator.Defaults.NumSummaryTermPerTopic,
            int numBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumBurninIterations,
            bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog), inputColumn, outputColumn, numTopic, alphaSum, beta, mhstep, numIterations, likelihoodInterval, numThreads, numMaxDocToken,
                numSummaryTermPerTopic, numBurninIterations, resetRandomGenerator);

        /// <!-- Badly formed XML comment ignored for member "M:Microsoft.ML.TextCatalog.LatentDirichletAllocation(Microsoft.ML.TransformsCatalog.TextTransforms,Microsoft.ML.Transforms.Text.LatentDirichletAllocationTransformer.ColumnInfo[])" -->
                        public static LatentDirichletAllocationEstimator LatentDirichletAllocation(this TransformsCatalog.TextTransforms catalog, params LatentDirichletAllocationTransformer.ColumnInfo[] columns)
            => new LatentDirichletAllocationEstimator(CatalogUtils.GetEnvironment(catalog), columns);
    }
}
