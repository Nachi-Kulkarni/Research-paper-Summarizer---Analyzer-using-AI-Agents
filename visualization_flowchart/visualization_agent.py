# Make sentence_transformers an optional dependency with fallback

try:
    from sentence_transformers import SentenceTransformer, util
    has_sentence_transformers = True
except ImportError:
    print("Warning: sentence_transformers not installed. Will use alternative domain detection method.")
    has_sentence_transformers = False

# agents/visualization_agent.py
"""Agent for generating advanced visualizations from research papers."""
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from config import OPENROUTER_MODEL as MODEL_NAME, DEFAULT_TEMPERATURE as TEMPERATURE, OPENROUTER_API_KEY
from langchain_openai import ChatOpenAI
import matplotlib
# Import re at module level to avoid "referenced before assignment" errors
import re
import base64
import io
import json
import traceback
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
import os
import random 
import time
import datetime     
import scipy

# Make seaborn TRULY optional
try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    print("Warning: seaborn not installed. Using matplotlib for visualizations.")
    has_seaborn = False
    # Define a minimal sns replacement with basic functions
    class SeabornReplacement:
        def set_style(self, *args, **kwargs):
            pass
        def set_palette(self, *args, **kwargs):
            pass
        def heatmap(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            plt.imshow(*args, **kwargs)
            return plt.gca()
        def lineplot(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            return plt.plot(*args, **kwargs)
        def scatterplot(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            return plt.scatter(*args, **kwargs)
        def barplot(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            x = kwargs.get('x', args[0] if args else None)
            y = kwargs.get('y', args[1] if len(args) > 1 else None)
            return plt.bar(x, y)
    sns = SeabornReplacement()

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.patches as mpatches # Added import

# Try to download NLTK resources safely
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTK resource download failed: {e}")

# Try to load spaCy model safely
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    nlp = None
    print(f"spaCy model loading failed: {e}")

# Create the LLM directly with OpenRouter config
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE
)

def extract_keywords(text, top_n=20):
    """Extract the most significant keywords from text using NLP techniques."""
    if not text:
        return []

    # Use spaCy if available for better keyword extraction
    if nlp:
        doc = nlp(text)
        # Extract meaningful entities and noun phrases
        keywords = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'EVENT', 'NORP', 'FAC', 'GPE', 'LOC']:
                keywords.append(ent.text.lower())

        # Add noun chunks for technical terms and concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 5:
                keywords.append(chunk.text.lower())

        # Count frequency
        keyword_freq = Counter(keywords)
        return [k for k, v in keyword_freq.most_common(top_n)]

    # Fallback to simpler approach if spaCy isn't available
    try:
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words and len(w) > 2]

        # Count word frequency
        word_freq = Counter(filtered_words)
        return [w for w, freq in word_freq.most_common(top_n)]
    except Exception as e:
        # Simple regex-based word count
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        return [w for w, freq in word_freq.most_common(top_n)]

DOMAIN_DESCRIPTIONS = {
    'computer_science': "Research in theoretical and applied computer science, including algorithms, data structures, software engineering, computer architecture, and programming languages.",
    'ai_ml': "Research in artificial intelligence and machine learning, covering neural networks, deep learning, natural language processing, computer vision, and robotics.",
    'ece': "Research in electrical and computer engineering, focusing on circuits, electronics, microcontrollers, signal processing, and embedded systems.",
    'ete': "Research in electronic and telecommunication engineering, dealing with communication systems, wireless technologies, signal processing, and network protocols.",
    'physics': "Research in physics, encompassing theoretical, experimental, and applied physics, including mechanics, electromagnetism, thermodynamics, quantum mechanics, and astrophysics.",
    'biology': "Research in biological sciences, covering molecular biology, genetics, cell biology, ecology, evolution, and bioinformatics.",
    'chemistry': "Research in chemistry, including organic, inorganic, physical, and analytical chemistry, material science, and chemical processes.",
    'mathematics': "Research in pure and applied mathematics, covering algebra, geometry, calculus, analysis, topology, statistics, and mathematical modeling.",
    'medicine': "Research in medical science and healthcare, including clinical studies, disease diagnosis and treatment, pharmacology, and public health.",
    'environmental_science': "Research in environmental science, focusing on climate change, ecology, pollution, conservation, and sustainability.",
    'social_science': "Research in social sciences, including sociology, psychology, economics, political science, and anthropology.",
    'materials_science': "Research in materials science and engineering, covering material properties, processing, characterization, and applications."
}

def detect_domain(paper_text: str, domain_descriptions=DOMAIN_DESCRIPTIONS) -> str:
    """Detect the domain of the research paper using available methods."""
    try:
        # Use sentence_transformers if available
        if has_sentence_transformers:
            model = SentenceTransformer('all-mpnet-base-v2')
            domain_names = list(domain_descriptions.keys())
            domain_desc_list = list(domain_descriptions.values())

            # Embed domain descriptions
            domain_embeddings = model.encode(domain_desc_list)

            # Embed paper text (using the first 3000 chars for efficiency)
            paper_embedding = model.encode(paper_text[:3000])

            # Calculate cosine similarities
            similarities = util.cos_sim(paper_embedding, domain_embeddings)[0]

            # Create domain scores dictionary
            domain_scores = {domain_names[i]: float(similarities[i]) for i in range(len(domain_names))}

            # Get top 2 domains
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            top_domain = sorted_domains[0][0]

            # If second domain is close to first, consider it a hybrid domain
            if len(sorted_domains) > 1:
                top_score = sorted_domains[0][1]
                second_score = sorted_domains[1][1]
                second_domain = sorted_domains[1][0]

                # If second domain score is at least 85% of top domain, it's a hybrid
                if top_score > 0 and second_score / top_score >= 0.85:
                    return f"{top_domain}/{second_domain}"

            return top_domain
        else:
            # Use alternative method (extract keywords and match against descriptions)
            keywords = extract_keywords(paper_text, top_n=100)
            
            # Count domain-related terms in the keywords
            domain_scores = {}
            for domain, description in domain_descriptions.items():
                # Split description into individual words and phrases
                domain_terms = description.lower().replace(',', ' ').replace('.', ' ').split()
                # Count matches between keywords and domain terms
                score = sum(1 for keyword in keywords if any(term in keyword for term in domain_terms))
                domain_scores[domain] = score
            
            # Get top domain
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            if not sorted_domains:
                return 'general_science'
                
            top_domain = sorted_domains[0][0]
            
            # Check for hybrid domain
            if len(sorted_domains) > 1:
                top_score = sorted_domains[0][1]
                second_score = sorted_domains[1][1]
                second_domain = sorted_domains[1][0]
                
                # If scores are close, consider it a hybrid domain
                if top_score > 0 and second_score / top_score >= 0.75:
                    return f"{top_domain}/{second_domain}"
            
            return top_domain
    except Exception as e:
        print(f"Domain detection error: {e}. Using general_science domain.")
        return "general_science"

def safe_execute_plotting_code(code: str, max_timeout: int = 30) -> list:
    """Execute matplotlib code in a restricted namespace and return base64 encoded images."""
    print("Executing visualization code...")
    print(f"Input code length: {len(code)} characters")
    
    # Clean code to remove markdown code blocks
    if "```" in code:
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if code_blocks:
            code = "\n".join(code_blocks)
        else:
            code = code.replace("```python", "").replace("```", "").strip()

    # Remove any bullet points or other markdown formatting
    code = code.replace("•python", "").replace("• python", "")

    # Fix gridspec import issue - two possible approaches
    if 'gridspec' in code:
        # Add explicit gridspec import at top of code
        if 'from matplotlib import gridspec' not in code and 'import matplotlib.gridspec' not in code:
            code = "import matplotlib.gridspec as gridspec\n" + code
        
        # Also ensure gridspec is correctly imported in the execution environment
        import matplotlib.gridspec
        gridspec = matplotlib.gridspec

    # Look for import statements and replace them with pre-loaded modules
    import_pattern = re.compile(r'^(?:import|from)\s+([a-zA-Z0-9_\.]+)', re.MULTILINE)
    imported_modules = import_pattern.findall(code)

    # Remove import statements as we'll pre-import everything
    code = re.sub(r'^(?:import|from)\s+[^\n]+', '# Import handled by safe execution environment', code, flags=re.MULTILINE)

    # Print modified code for debugging
    print("Modified code (imports handled):")
    print(code[:200] + "..." if len(code) > 200 else code)

    # Check if code is valid before executing
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        code = """
# Creating a fallback visualization due to syntax error
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Fallback Visualization - Syntax Error')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
"""

    # Create a restricted namespace with essential functions and modules
    safe_builtins = {
        'range': range,
        'len': len,
        'int': int,
        'float': float,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'min': min,
        'max': max,
        'sum': sum,
        'abs': abs,
        'round': round,
        'print': print,
        'zip': zip,
        'enumerate': enumerate,
        'sorted': sorted,
        'filter': filter,
        'map': map,
        'any': any,
        'all': all,
        'divmod': divmod,
        'pow': pow,
        'complex': complex,
        'bool': bool,
        'set': set,
        'frozenset': frozenset,
        'slice': slice,
        'iter': iter,
        'next': next,
        'reversed': reversed,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'hasattr': hasattr,
        'getattr': getattr,
        'setattr': setattr,
        'callable': callable,
        '__import__': __import__ # Include __import__ to allow dynamic imports (with caution)
    }

    # Import required modules outside the execution environment
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import io
    import base64
    import math

    # Pre-import scipy modules
    try:
        import scipy.signal as signal
        import scipy.stats as stats
        from scipy import interpolate
        from scipy.cluster import hierarchy
        import scipy.fft as fft # Added fft module
        import scipy.optimize as optimize # Added optimize module
        import scipy.constants as constants # Added constants module
        import scipy.special as special # Added special functions
        import scipy.ndimage as ndimage # Added ndimage
        import scipy.integrate as integrate # Added integrate
        import scipy.linalg as linalg # Added linalg
    except ImportError:
        signal = None
        stats = None
        interpolate = None
        hierarchy = None
        fft = None
        optimize = None
        constants = None
        special = None
        ndimage = None
        integrate = None
        linalg = None


    try:
        from sklearn import (
            decomposition,
            datasets,
            cluster,
            metrics,
            preprocessing,
            manifold,
            linear_model, # Added linear_model
            svm, # Added svm
            tree, # Added tree
            ensemble, # Added ensemble
            neighbors, # Added neighbors
            naive_bayes, # Added naive_bayes
            discriminant_analysis, # Added discriminant_analysis
            gaussian_process # Added gaussian_process
        )
    except ImportError:
        decomposition = None
        datasets = None
        cluster = None
        metrics = None
        preprocessing = None
        manifold = None
        linear_model = None
        svm = None
        tree = None
        ensemble = None
        neighbors = None
        naive_bayes = None
        discriminant_analysis = None
        gaussian_process = None


    try:
        import pandas as pd
    except ImportError:
        pd = None

    # Define local variables with properly imported modules
    local_vars = {
        'plt': plt,
        'np': np,
        'matplotlib': matplotlib,
        'patches': mpatches, # Added mpatches to local_vars as 'patches'
        'io': io,
        'base64': base64,
        'math': math,
        'signal': signal,
        'stats': stats,
        'interpolate': interpolate,
        'pd': pd,
        'sns': sns,  # This will be None if not installed
        'GridSpec': GridSpec,
        'MaxNLocator': MaxNLocator,
        'gridspec': matplotlib.gridspec,  # Add explicit gridspec module
        're': re,  # Add re module to support regex operations
        'os': os,  # Add os module for path operations
        'random': __import__('random'),  # Add random module
        'fft': fft,
        'optimize': optimize,
        'constants': constants,
        'special': special,
        'ndimage': ndimage,
        'integrate': integrate,
        'linalg': linalg
    }

    # Add sklearn modules if available
    if decomposition is not None:
        local_vars['decomposition'] = decomposition
        local_vars['PCA'] = decomposition.PCA if hasattr(decomposition, 'PCA') else None
        local_vars['NMF'] = decomposition.NMF if hasattr(decomposition, 'NMF') else None
        local_vars['TruncatedSVD'] = decomposition.TruncatedSVD if hasattr(decomposition, 'TruncatedSVD') else None

    if manifold is not None:
        local_vars['manifold'] = manifold
        local_vars['TSNE'] = manifold.TSNE if hasattr(manifold, 'TSNE') else None
        local_vars['MDS'] = manifold.MDS if hasattr(manifold, 'MDS') else None
        local_vars['LocallyLinearEmbedding'] = manifold.LocallyLinearEmbedding if hasattr(manifold, 'LocallyLinearEmbedding') else None
        local_vars['SpectralEmbedding'] = manifold.SpectralEmbedding if hasattr(manifold, 'SpectralEmbedding') else None

    if datasets is not None:
        local_vars['datasets'] = datasets
        local_vars['make_classification'] = datasets.make_classification if hasattr(datasets, 'make_classification') else None
        local_vars['make_blobs'] = datasets.make_blobs if hasattr(datasets, 'make_blobs') else None
        local_vars['make_regression'] = datasets.make_regression if hasattr(datasets, 'make_regression') else None
        local_vars['load_iris'] = datasets.load_iris if hasattr(datasets, 'load_iris') else None
        local_vars['load_digits'] = datasets.load_digits if hasattr(datasets, 'load_digits') else None
        local_vars['load_wine'] = datasets.load_wine if hasattr(datasets, 'load_wine') else None
        local_vars['fetch_olivetti_faces'] = datasets.fetch_olivetti_faces if hasattr(datasets, 'fetch_olivetti_faces') else None
        local_vars['load_sample_image'] = datasets.load_sample_image if hasattr(datasets, 'load_sample_image') else None


    if cluster is not None:
        local_vars['cluster'] = cluster
        local_vars['KMeans'] = cluster.KMeans if hasattr(cluster, 'KMeans') else None
        local_vars['DBSCAN'] = cluster.DBSCAN if hasattr(cluster, 'DBSCAN') else None
        local_vars['AgglomerativeClustering'] = cluster.AgglomerativeClustering if hasattr(cluster, 'AgglomerativeClustering') else None
        local_vars['SpectralClustering'] = cluster.SpectralClustering if hasattr(cluster, 'SpectralClustering') else None
        local_vars['MiniBatchKMeans'] = cluster.MiniBatchKMeans if hasattr(cluster, 'MiniBatchKMeans') else None
        local_vars['MeanShift'] = cluster.MeanShift if hasattr(cluster, 'MeanShift') else None
        local_vars['AffinityPropagation'] = cluster.AffinityPropagation if hasattr(cluster, 'AffinityPropagation') else None
        local_vars['Birch'] = cluster.Birch if hasattr(cluster, 'Birch') else None

    if metrics is not None:
        local_vars['metrics'] = metrics
        local_vars['confusion_matrix'] = metrics.confusion_matrix if hasattr(metrics, 'confusion_matrix') else None
        local_vars['accuracy_score'] = metrics.accuracy_score if hasattr(metrics, 'accuracy_score') else None
        local_vars['precision_score'] = metrics.precision_score if hasattr(metrics, 'precision_score') else None
        local_vars['recall_score'] = metrics.recall_score if hasattr(metrics, 'recall_score') else None
        local_vars['f1_score'] = metrics.f1_score if hasattr(metrics, 'f1_score') else None
        local_vars['roc_curve'] = metrics.roc_curve if hasattr(metrics, 'roc_curve') else None
        local_vars['auc'] = metrics.auc if hasattr(metrics, 'auc') else None
        local_vars['classification_report'] = metrics.classification_report if hasattr(metrics, 'classification_report') else None
        local_vars['mean_squared_error'] = metrics.mean_squared_error if hasattr(metrics, 'mean_squared_error') else None
        local_vars['r2_score'] = metrics.r2_score if hasattr(metrics, 'r2_score') else None
        local_vars['silhouette_score'] = metrics.silhouette_score if hasattr(metrics, 'silhouette_score') else None
        local_vars['calinski_harabasz_score'] = metrics.calinski_harabasz_score if hasattr(metrics, 'calinski_harabasz_score') else None
        local_vars['davies_bouldin_score'] = metrics.davies_bouldin_score if hasattr(metrics, 'davies_bouldin_score') else None


    if preprocessing is not None:
        local_vars['preprocessing'] = preprocessing
        local_vars['StandardScaler'] = preprocessing.StandardScaler if hasattr(preprocessing, 'StandardScaler') else None
        local_vars['MinMaxScaler'] = preprocessing.MinMaxScaler if hasattr(preprocessing, 'MinMaxScaler') else None
        local_vars['RobustScaler'] = preprocessing.RobustScaler if hasattr(preprocessing, 'RobustScaler') else None
        local_vars['Normalizer'] = preprocessing.Normalizer if hasattr(preprocessing, 'Normalizer') else None
        local_vars['LabelEncoder'] = preprocessing.LabelEncoder if hasattr(preprocessing, 'LabelEncoder') else None
        local_vars['OneHotEncoder'] = preprocessing.OneHotEncoder if hasattr(preprocessing, 'OneHotEncoder') else None
        local_vars['PolynomialFeatures'] = preprocessing.PolynomialFeatures if hasattr(preprocessing, 'PolynomialFeatures') else None
        local_vars['PowerTransformer'] = preprocessing.PowerTransformer if hasattr(preprocessing, 'PowerTransformer') else None
        local_vars['QuantileTransformer'] = preprocessing.QuantileTransformer if hasattr(preprocessing, 'QuantileTransformer') else None
        local_vars['Binarizer'] = preprocessing.Binarizer if hasattr(preprocessing, 'Binarizer') else None
        local_vars['KBinsDiscretizer'] = preprocessing.KBinsDiscretizer if hasattr(preprocessing, 'KBinsDiscretizer') else None


    # Add sklearn models if available
    if linear_model is not None:
        local_vars['linear_model'] = linear_model
        local_vars['LinearRegression'] = linear_model.LinearRegression if hasattr(linear_model, 'LinearRegression') else None
        local_vars['LogisticRegression'] = linear_model.LogisticRegression if hasattr(linear_model, 'LogisticRegression') else None
        local_vars['Ridge'] = linear_model.Ridge if hasattr(linear_model, 'Ridge') else None
        local_vars['Lasso'] = linear_model.Lasso if hasattr(linear_model, 'Lasso') else None
        local_vars['ElasticNet'] = linear_model.ElasticNet if hasattr(linear_model, 'ElasticNet') else None
        local_vars['SGDRegressor'] = linear_model.SGDRegressor if hasattr(linear_model, 'SGDRegressor') else None
        local_vars['SGDClassifier'] = linear_model.SGDClassifier if hasattr(linear_model, 'SGDClassifier') else None
        local_vars['Perceptron'] = linear_model.Perceptron if hasattr(linear_model, 'Perceptron') else None

    if svm is not None:
        local_vars['svm'] = svm
        local_vars['SVC'] = svm.SVC if hasattr(svm, 'SVC') else None
        local_vars['SVR'] = svm.SVR if hasattr(svm, 'SVR') else None
        local_vars['LinearSVC'] = svm.LinearSVC if hasattr(svm, 'LinearSVC') else None
        local_vars['LinearSVR'] = svm.LinearSVR if hasattr(svm, 'LinearSVR') else None
        local_vars['NuSVC'] = svm.NuSVC if hasattr(svm, 'NuSVC') else None
        local_vars['NuSVR'] = svm.NuSVR if hasattr(svm, 'NuSVR') else None
        local_vars['OneClassSVM'] = svm.OneClassSVM if hasattr(svm, 'OneClassSVM') else None

    if tree is not None:
        local_vars['tree'] = tree
        local_vars['DecisionTreeClassifier'] = tree.DecisionTreeClassifier if hasattr(tree, 'DecisionTreeClassifier') else None
        local_vars['DecisionTreeRegressor'] = tree.DecisionTreeRegressor if hasattr(tree, 'DecisionTreeRegressor') else None
        local_vars['ExtraTreeClassifier'] = tree.ExtraTreeClassifier if hasattr(tree, 'ExtraTreeClassifier') else None
        local_vars['ExtraTreeRegressor'] = tree.ExtraTreeRegressor if hasattr(tree, 'ExtraTreeRegressor') else None

    if ensemble is not None:
        local_vars['ensemble'] = ensemble
        local_vars['RandomForestClassifier'] = ensemble.RandomForestClassifier if hasattr(ensemble, 'RandomForestClassifier') else None
        local_vars['RandomForestRegressor'] = ensemble.RandomForestRegressor if hasattr(ensemble, 'RandomForestRegressor') else None
        local_vars['GradientBoostingClassifier'] = ensemble.GradientBoostingClassifier if hasattr(ensemble, 'GradientBoostingClassifier') else None
        local_vars['GradientBoostingRegressor'] = ensemble.GradientBoostingRegressor if hasattr(ensemble, 'GradientBoostingRegressor') else None
        local_vars['AdaBoostClassifier'] = ensemble.AdaBoostClassifier if hasattr(ensemble, 'AdaBoostClassifier') else None
        local_vars['AdaBoostRegressor'] = ensemble.AdaBoostRegressor if hasattr(ensemble, 'AdaBoostRegressor') else None
        local_vars['HistGradientBoostingClassifier'] = ensemble.HistGradientBoostingClassifier if hasattr(ensemble, 'HistGradientBoostingClassifier') else None
        local_vars['HistGradientBoostingRegressor'] = ensemble.HistGradientBoostingRegressor if hasattr(ensemble, 'HistGradientBoostingRegressor') else None
        local_vars['BaggingClassifier'] = ensemble.BaggingClassifier if hasattr(ensemble, 'BaggingClassifier') else None
        local_vars['BaggingRegressor'] = ensemble.BaggingRegressor if hasattr(ensemble, 'BaggingRegressor') else None
        local_vars['ExtraTreesClassifier'] = ensemble.ExtraTreesClassifier if hasattr(ensemble, 'ExtraTreesClassifier') else None
        local_vars['ExtraTreesRegressor'] = ensemble.ExtraTreesRegressor if hasattr(ensemble, 'ExtraTreesRegressor') else None
        local_vars['VotingClassifier'] = ensemble.VotingClassifier if hasattr(ensemble, 'VotingClassifier') else None
        local_vars['VotingRegressor'] = ensemble.VotingRegressor if hasattr(ensemble, 'VotingRegressor') else None
        local_vars['StackingClassifier'] = ensemble.StackingClassifier if hasattr(ensemble, 'StackingClassifier') else None
        local_vars['StackingRegressor'] = ensemble.StackingRegressor if hasattr(ensemble, 'StackingRegressor') else None

    if neighbors is not None:
        local_vars['neighbors'] = neighbors
        local_vars['KNeighborsClassifier'] = neighbors.KNeighborsClassifier if hasattr(neighbors, 'KNeighborsClassifier') else None
        local_vars['KNeighborsRegressor'] = neighbors.KNeighborsRegressor if hasattr(neighbors, 'KNeighborsRegressor') else None
        local_vars['RadiusNeighborsClassifier'] = neighbors.RadiusNeighborsClassifier if hasattr(neighbors, 'RadiusNeighborsClassifier') else None
        local_vars['RadiusNeighborsRegressor'] = neighbors.RadiusNeighborsRegressor if hasattr(neighbors, 'RadiusNeighborsRegressor') else None
        local_vars['NearestNeighbors'] = neighbors.NearestNeighbors if hasattr(neighbors, 'NearestNeighbors') else None
        local_vars['KNeighborsTransformer'] = neighbors.KNeighborsTransformer if hasattr(neighbors, 'KNeighborsTransformer') else None
        local_vars['RadiusNeighborsTransformer'] = neighbors.RadiusNeighborsTransformer if hasattr(neighbors, 'RadiusNeighborsTransformer') else None

    if naive_bayes is not None:
        local_vars['naive_bayes'] = naive_bayes
        local_vars['GaussianNB'] = naive_bayes.GaussianNB if hasattr(naive_bayes, 'GaussianNB') else None
        local_vars['MultinomialNB'] = naive_bayes.MultinomialNB if hasattr(naive_bayes, 'MultinomialNB') else None
        local_vars['ComplementNB'] = naive_bayes.ComplementNB if hasattr(naive_bayes, 'ComplementNB') else None
        local_vars['BernoulliNB'] = naive_bayes.BernoulliNB if hasattr(naive_bayes, 'BernoulliNB') else None
        local_vars['CategoricalNB'] = naive_bayes.CategoricalNB if hasattr(naive_bayes, 'CategoricalNB') else None

    if discriminant_analysis is not None:
        local_vars['discriminant_analysis'] = discriminant_analysis
        local_vars['LinearDiscriminantAnalysis'] = discriminant_analysis.LinearDiscriminantAnalysis if hasattr(discriminant_analysis, 'LinearDiscriminantAnalysis') else None
        local_vars['QuadraticDiscriminantAnalysis'] = discriminant_analysis.QuadraticDiscriminantAnalysis if hasattr(discriminant_analysis, 'QuadraticDiscriminantAnalysis') else None

    if gaussian_process is not None:
        local_vars['gaussian_process'] = gaussian_process
        local_vars['GaussianProcessClassifier'] = gaussian_process.GaussianProcessClassifier if hasattr(gaussian_process, 'GaussianProcessClassifier') else None
        local_vars['GaussianProcessRegressor'] = gaussian_process.GaussianProcessRegressor if hasattr(gaussian_process, 'GaussianProcessRegressor') else None


    # Import additional modules that might be needed
    for module_name in imported_modules:
        module_base = module_name.split('.')[0]
        if module_base not in local_vars and module_base not in ['matplotlib', 'numpy', 'scipy', 'pandas', 'seaborn', 'sklearn']:
            try:
                # Try to safely import the requested module
                module = __import__(module_base)
                local_vars[module_base] = module
                print(f"Imported {module_base} module")
            except ImportError:
                print(f"Warning: Could not import {module_base}")

    images = []

    # Import signal handling for timeout
    import signal as signal_module

    # Define timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Visualization code execution timed out after {max_timeout} seconds")

    # Initialize globals_dict at the top level to ensure it's always defined
    globals_dict = {
        '__builtins__': safe_builtins,
        'np': np,
        'plt': plt,
        'pd': pd,
        'math': math,
        'random': random,
        're': re,
        'os': os,
        'datetime': datetime,
        'scipy': scipy,
        '__name__': "__main__",
        'print': print
    }
    
    # Safer dictionary access function to avoid KeyError
    def safe_get(dictionary, key, default=None):
        """Safely access dictionary keys without KeyError."""
        return dictionary.get(key, default)
        
    # Add safe_get to the globals
    globals_dict['safe_get'] = safe_get

    try:
        # Set up the timeout
        signal_module.signal(signal_module.SIGALRM, timeout_handler)
        signal_module.alarm(max_timeout)

        # FIXED: Combine safe_builtins and local_vars properly
        # Create a globals dictionary that includes both builtins and module imports
        globals_dict.update(local_vars)  # Add all the local variables to globals

        # Add defensive code before execution that handles common errors
        try:
            # Check for KeyError with 'Top-1 Error' specifically
            if "Top-1 Error" in code and "'Top-1 Error'" in code:
                # Add protective code at the beginning
                code = """
# Add defensive programming to avoid KeyError
def safe_get(d, key, default=None):
    return d.get(key, default)

# Replace dictionary[key] with safe_get(dictionary, key) for problematic keys
""" + code
                # Replace direct dictionary access with safe_get for problematic keys
                code = code.replace("['Top-1 Error']", ".get('Top-1 Error', 0)")
                code = code.replace("['Top-5 Error']", ".get('Top-5 Error', 0)")
                code = code.replace("['Accuracy']", ".get('Accuracy', 0)")
                
                print("Applied defensive programming to avoid KeyError")
            
            # Add general index error protection
            index_error_protection = """
# Add defensive programming for list index access
def safe_index(lst, idx, default=None):
    try:
        if isinstance(lst, (list, tuple)) and 0 <= idx < len(lst):
            return lst[idx]
        return default
    except:
        return default
"""
            code = index_error_protection + code
            
            # Replace common patterns that might cause index errors
            code = code.replace("data[0]", "safe_index(data, 0)")
            code = code.replace("labels[0]", "safe_index(labels, 0)")
            code = code.replace("colors[0]", "safe_index(colors, 0)")
            
            print("Applied defensive programming to avoid IndexError")
            
            # Fix for "ValueError: keyword ha is not recognized" in ax3.tick_params
            # This replaces the specific line generated by the LLM that causes the error.
            original_tick_params_line = "ax3.tick_params(axis='x', rotation=45, ha='right', fontsize=10)"
            corrected_tick_params_line = "ax3.tick_params(axis='x', rotation=45, fontsize=10)"
            if original_tick_params_line in code:
                code = code.replace(original_tick_params_line, corrected_tick_params_line)
                print("Applied specific fix for 'ha' keyword in ax3.tick_params.")
            
        except Exception as preprocess_error:
            print(f"Warning: Error in preprocessing code: {preprocess_error}")
            
        # Execute the code with the combined globals dictionary and an empty locals dictionary
        exec(code, globals_dict)

        # Turn off the alarm
        signal_module.alarm(0)

        # Check if there are any figures to save
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                # Save the figure to a bytes buffer
                buf = io.BytesIO()
                # Ensure output directory exists
                output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'diagrams')
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory: {output_dir}")
                
                # Save to file with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f'visualization_{timestamp}_{fig_num}.png')
                fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
                print(f"Saved visualization to: {output_path}")
                
                # Also save to buffer for base64 encoding
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)

                # Encode the image as base64
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                images.append(img_str)

                # Close the figure to free memory
                plt.close(fig)
        else:
            print("No figures were created by the LLM-generated code.")
            # Removed create_default_visualization() and subsequent saving/appending
    except TimeoutError as te:
        print(f"Timeout error: {te}")
        create_error_visualization(f"Execution timed out after {max_timeout} seconds. The visualization code was too complex or contained infinite loops.")

        # Save the error figure
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(img_str)

        # Close the figure
        plt.close(fig)
    except KeyError as ke:
        print(f"KeyError in visualization code: {ke}")
        # Create a visualization showing the specific key error
        create_error_visualization(f"KeyError: The key '{ke}' was not found in the dictionary.\n\n"
                                  f"This typically happens when the code assumes data contains certain keys.\n"
                                  f"Check the data structure or add error handling for missing keys.")

        # Save the error figure
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(img_str)

        # Close the figure
        plt.close(fig)
    except IndexError as idx_err:
        print(f"IndexError in visualization code: {idx_err}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        
        # Create a visualization showing the specific index error
        create_error_visualization(f"IndexError: {str(idx_err)}\n\n"
                                  f"This typically happens when trying to access a list element that doesn't exist.\n"
                                  f"Check array sizes and indices in the code.")

        # Save the error figure
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(img_str)

        # Close the figure
        plt.close(fig)
    except ValueError as ve:
        print(f"ValueError in visualization code: {ve}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")

        # Create a visualization showing the specific value error
        create_error_visualization(f"ValueError: {str(ve)}\n\n"
                                  f"Check the parameters used in plotting functions.\n"
                                  f"Traceback:\n{traceback_str[:500]}")

        # Save the error figure
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(img_str)

        # Close the figure
        plt.close(fig)
    except Exception as e:
        print(f"Error in safe_execute_plotting_code: {e}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")

        # Create a figure showing the error
        create_error_visualization(f"{str(e)}\n\nTraceback:\n{traceback_str[:500]}")

        # Save the error figure
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Encode the image as base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(img_str)

        # Close the figure
        plt.close(fig)
    finally:
        # Ensure alarm is turned off
        signal_module.alarm(0)

    return images

def create_default_visualization():
    """Create a sophisticated default visualization with multiple visualization types"""
    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=plt.gcf())

    # Generate placeholder data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x/2)

    # Add random noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, y1.shape)
    y1_noisy = y1 + noise

    # Create multi-panel visualization
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(x, y1, 'b-', label='Signal')
    ax1.plot(x, y1_noisy, 'r.', alpha=0.3, label='Noisy data')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_title('Time Series Data')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Create 2D visualization with colormap
    ax2 = plt.subplot(gs[0, 1:])
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = np.sin(xx) * np.cos(yy) * np.exp(-(xx**2 + yy**2)/10)
    im = ax2.pcolormesh(xx, yy, zz, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax2, label='Value')
    ax2.set_xlabel('X dimension')
    ax2.set_ylabel('Y dimension')
    ax2.set_title('2D Field Visualization')

    # Create bar chart
    ax3 = plt.subplot(gs[1, 0])
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.rand(5) * 10
    bars = ax3.bar(categories, values, color=plt.cm.Blues(np.linspace(0.4, 0.8, 5)))
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Value')
    ax3.set_title('Categorical Data')
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add error bars
    errors = values * np.random.rand(5) * 0.3
    for i, (bar, err) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + err + 0.1,
                f'{height:.1f}', ha='center', va='bottom', rotation=0)
        ax3.errorbar(i, height, yerr=err, fmt='none', ecolor='black', capsize=5)

    # Create scatter plot with regression
    ax4 = plt.subplot(gs[1, 1])
    # Generate correlated data
    n = 50
    x_scatter = np.random.rand(n) * 10
    y_scatter = 2 + 0.5 * x_scatter + np.random.normal(0, 1, n)

    # Fit linear regression
    m, b = np.polyfit(x_scatter, y_scatter, 1)
    x_fit = np.array([0, 10])
    y_fit = m * x_fit + b

    # Plot data and fit
    ax4.scatter(x_scatter, y_scatter, alpha=0.7, c=y_scatter, cmap='plasma')
    ax4.plot(x_fit, y_fit, 'r-', label=f'y = {m:.2f}x + {b:.2f}')
    ax4.set_xlabel('Independent variable')
    ax4.set_ylabel('Dependent variable')
    ax4.set_title('Regression Analysis')
    ax4.legend()

    # Create info panel
    ax5 = plt.subplot(gs[1, 2])
    ax5.text(0.5, 0.5, 'Paper-Specific Visualization\n\nCustomized visualizations are\ngenerated based on the\ncontent and domain of\nthe research paper.\n\nThe agent analyzes key concepts\nand generates relevant graphics\nto illustrate research findings.',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    ax5.axis('off')

    plt.tight_layout()
    plt.suptitle('Advanced Research Visualization', fontsize=16, y=0.98)

def create_error_visualization(error_message):
    """Create an enhanced visualization showing an error message with debugging information"""
    plt.figure(figsize=(10, 6))

    # Create a grid for better layout
    gs = GridSpec(3, 1, height_ratios=[1, 2, 1], hspace=0.4)

    # Title area
    ax_title = plt.subplot(gs[0])
    ax_title.text(0.5, 0.5, "Visualization Error",
                  ha='center', va='center', fontsize=18, color='darkred',
                  weight='bold')
    ax_title.axis('off')

    # Add a red border to indicate error
    ax_error = plt.subplot(gs[1])
    ax_error.spines['bottom'].set_color('red')
    ax_error.spines['top'].set_color('red')
    ax_error.spines['left'].set_color('red')
    ax_error.spines['right'].set_color('red')
    ax_error.spines['bottom'].set_linewidth(2)
    ax_error.spines['top'].set_linewidth(2)
    ax_error.spines['left'].set_linewidth(2)
    ax_error.spines['right'].set_linewidth(2)

    # Add error message with better formatting
    error_text = f"Error in visualization generation:\n\n{error_message}"
    ax_error.text(0.5, 0.5, error_text,
                 ha='center', va='center', fontsize=10,
                 wrap=True, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ffeeee", ec="red", alpha=0.9))
    ax_error.axis('off')

    # Add suggestion area
    ax_suggest = plt.subplot(gs[2])
    suggest_text = "The system will generate an alternative visualization.\n" + \
                   "Please check the error message for debugging information."
    ax_suggest.text(0.5, 0.5, suggest_text,
                   ha='center', va='center', fontsize=10, style='italic', color='gray',
                   bbox=dict(boxstyle="round,pad=0.5", fc="#eeeeee", ec="gray", alpha=0.8))
    ax_suggest.axis('off')

def extract_numerical_data(paper_text):
    """Extract any potential numerical data from the paper text."""
    # Enhanced patterns to capture more numerical data formats
    percentage_pattern = r'(\d+(?:\.\d+)?)%'
    measurement_pattern = r'(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)'
    value_range_pattern = r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)' # e.g., "5-10", "2.5 - 3.7"
    ratio_pattern = r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)' # e.g., "1:2", "3.5:1"
    single_number_pattern = r'\b(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\b' # Captures numbers with commas and decimals, up to 4 digits before comma


    percentages = [float(p) for p in re.findall(percentage_pattern, paper_text)]
    measurements = [(float(m[0]), float(m[1])) for m in re.findall(measurement_pattern, paper_text)]
    value_ranges = [(float(r[0]), float(r[1])) for r in re.findall(value_range_pattern, paper_text)]
    ratios = [(float(ratio[0]), float(ratio[1])) for ratio in re.findall(ratio_pattern, paper_text)]

    # Extract sequences of numbers and table-like structures more robustly
    number_sequences = re.findall(r'(\d+(?:\.\d+)?(?:,\s+|\s+)\d+(?:\.\d+)?(?:,\s+|\s+)\d+(?:\.\d+)?)', paper_text) # At least 3 numbers in sequence
    number_sequences = [list(map(float, seq.replace(',', ' ').split())) for seq in number_sequences] # Handle comma or space

    table_rows = re.findall(r'(\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?){2,})', paper_text) # At least 3 numbers in a row
    table_rows = [list(map(float, row.split())) for row in table_rows]


    # General number extraction (less structured, for fallback)
    all_numbers = [float(num.replace(',', '')) for num in re.findall(single_number_pattern, paper_text)] # Remove commas from numbers like "1,000"

    # Package the extracted data
    data = {
        'percentages': percentages,
        'measurements': measurements,
        'value_ranges': value_ranges,
        'ratios': ratios,
        'table_rows': table_rows,
        'number_sequences': number_sequences,
        'all_numbers': all_numbers # Added general numbers
    }

    return data

def are_values_similar(values, threshold_percentage=0.1):
    """Check if values are considered similar based on their range."""
    if not values:
        return False
    val_range = np.max(values) - np.min(values)
    max_val = np.max(np.abs(values)) # Use max absolute value to avoid issues with negative numbers
    if max_val == 0: # Avoid division by zero if all values are zero
        return True
    return (val_range / max_val) < threshold_percentage

def domain_specific_visualization(domain, extracted_data=None):
    """Create domain-specific fallback visualizations when LLM generation fails."""
    # Ensure extracted_data is not None to prevent NoneType errors
    if extracted_data is None:
        extracted_data = {
            'percentages': [],
            'measurements': [],
            'table_rows': [],
            'number_sequences': [],
            'value_ranges': [], # Added value_ranges
            'ratios': [], # Added ratios
            'all_numbers': [] # Added all_numbers
        }

    plt.figure(figsize=(12, 8))

    if domain == 'ai_ml' or domain.startswith('ai_ml/'):
        # AI/ML visualization (remained mostly the same, could be further generalized if needed)
        ax1 = plt.subplot(2, 2, 1)
        layer_sizes = [4, 8, 12, 8, 4, 1]
        layer_positions = []
        max_size = max(layer_sizes)
        for i, size in enumerate(layer_sizes):
            positions = []
            for j in range(size):
                positions.append((i, j * (max_size - 1) / (size - 1 if size > 1 else 1)))
            layer_positions.append(positions)
        for i, layer in enumerate(layer_positions):
            for x, y in layer:
                color = 'green' if i == 0 else ('red' if i == len(layer_positions) - 1 else 'blue')
                ax1.scatter(x, y, s=100, color=color, edgecolors='black', zorder=2)
        for i in range(len(layer_positions) - 1):
            for j, (x1, y1) in enumerate(layer_positions[i]):
                for k, (x2, y2) in enumerate(layer_positions[i + 1]):
                    alpha = 0.3 if np.random.rand() > 0.7 else 0.1
                    ax1.plot([x1, x2], [y1, y2], 'k-', alpha=alpha, zorder=1)
        ax1.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax1.set_ylim(-0.5, max_size - 0.5)
        ax1.set_title('Neural Network Architecture')
        ax1.axis('off')

        ax2 = plt.subplot(2, 2, 2)
        epochs = np.arange(1, 101)
        train_loss = 1 - 1/(1 + np.exp(-epochs/20))
        val_loss = 1 - 1/(1 + np.exp(-epochs/25)) + 0.1 * np.exp(-epochs/10)
        ax2.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = plt.subplot(2, 2, 3)
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
        importances = np.array([0.35, 0.25, 0.18, 0.12, 0.1])
        importances = importances / np.sum(importances)
        ax3.pie(importances, labels=features, autopct='%1.1f%%', startangle=90, colors=plt.cm.viridis(importances))
        ax3.axis('equal')
        ax3.set_title('Feature Importance (Pie Chart)')

        ax4 = plt.subplot(2, 2, 4)
        conf_matrix = np.array([[0.9, 0.06, 0.04], [0.05, 0.85, 0.1], [0.02, 0.08, 0.9]])
        im = ax4.imshow(conf_matrix, cmap='Blues')
        ax4.set_xticks(np.arange(3))
        ax4.set_yticks(np.arange(3))
        ax4.set_xticklabels(['Class A', 'Class B', 'Class C'])
        ax4.set_yticklabels(['Class A', 'Class B', 'Class C'])
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
        ax4.set_title('Confusion Matrix')
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, f'{conf_matrix[i, j]:.2f}', ha='center', va='center', color='white' if conf_matrix[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax4, label='Probability')

    elif domain == 'computer_science' or domain.startswith('computer_science/'):
        # Computer Science visualization (remained mostly the same, can be generalized further)
        ax1 = plt.subplot(2, 2, 1)
        n = np.linspace(1, 100, 100)
        constant = np.ones_like(n)
        logarithmic = np.log(n)
        linear = n
        linearithmic = n * np.log(n)
        quadratic = n**2
        cubic = n**3
        exponential = 2**n / 2**90
        ax1.plot(n, constant, 'k-', label='O(1)', linewidth=2)
        ax1.plot(n, logarithmic, 'g-', label='O(log n)', linewidth=2)
        ax1.plot(n, linear, 'b-', label='O(n)', linewidth=2)
        ax1.plot(n, linearithmic, 'c-', label='O(n log n)', linewidth=2)
        ax1.plot(n, quadratic, 'y-', label='O(n²)', linewidth=2)
        ax1.plot(n, cubic, 'r-', label='O(n³)', linewidth=2)
        ax1.plot(n, exponential, 'm-', label='O(2ⁿ)', linewidth=2)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 150)
        ax1.set_xlabel('Input Size (n)')
        ax1.set_ylabel('Time Complexity')
        ax1.set_title('Algorithm Complexity Classes')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        data_structures = ['Array', 'Linked List', 'Hash Table', 'BST', 'Heap']
        operations = ['Access', 'Search', 'Insert', 'Delete']
        scores = np.array([[1, 3, 3, 3], [3, 3, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [1, 3, 1, 2]])
        im = ax2.imshow(scores, cmap='YlGnBu_r')
        ax2.set_xticks(np.arange(len(operations)))
        ax2.set_yticks(np.arange(len(data_structures)))
        ax2.set_xticklabels(operations)
        ax2.set_yticklabels(data_structures)
        complexity_labels = {1: 'O(1)', 2: 'O(log n)', 3: 'O(n)', 4: 'O(n log n)'}
        for i in range(len(data_structures)):
            for j in range(len(operations)):
                ax2.text(j, i, complexity_labels[scores[i, j]], ha='center', va='center')
        ax2.set_title('Data Structure Operation Complexities (Heatmap)')

        ax3 = plt.subplot(2, 2, 3)
        try:
            import networkx as nx
            G = nx.DiGraph()
        except ImportError:
            nx = None
            G = None

        if G:
            G.add_nodes_from(range(1, 9))
            edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (5, 8), (6, 8), (7, 8)]
            G.add_edges_from(edges)
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, edgecolors='black', ax=ax3)
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', ax=ax3)
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax3)
        else:
            nodes = np.array([[0.1, 0.9], [0.3, 0.7], [0.3, 0.3], [0.5, 0.8], [0.5, 0.5], [0.5, 0.2], [0.7, 0.7], [0.8, 0.4]])
            edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (4, 7), (5, 7), (6, 7)]
            ax3.scatter(nodes[:, 0], nodes[:, 1], s=500, color='lightblue', edgecolors='black', zorder=2)
            for i, j in edges:
                ax3.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'k-', alpha=0.6, zorder=1)
            for i in range(len(nodes)):
                ax3.text(nodes[i, 0], nodes[i, 1], str(i+1), ha='center', va='center', fontsize=10)
        ax3.set_title('Graph Data Structure')
        ax3.axis('off')

        ax4 = plt.subplot(2, 2, 4)
        algorithms = ['Alg A', 'Alg B', 'Alg C', 'Alg D']
        small_input = np.array([10, 15, 5, 8])
        medium_input = np.array([20, 35, 12, 25])
        large_input = np.array([50, 90, 35, 70])
        execution_times = [small_input, medium_input, large_input]
        input_sizes = ['Small', 'Medium', 'Large']
        x_pos = np.arange(len(input_sizes))
        colors = ['blue', 'green', 'red', 'purple']
        for i, algo_times in enumerate(zip(*execution_times)):
            ax4.plot(x_pos, algo_times, marker='o', linestyle='-', color=colors[i], label=algorithms[i])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(input_sizes)
        ax4.set_ylabel('Execution Time (ms)')
        ax4.set_title('Algorithm Performance Comparison (Line Plot)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')


    elif domain == 'ece' or domain.startswith('ece/'):
        # ECE visualization (remained mostly the same, can be generalized further)
        ax1 = plt.subplot(2, 2, 1)
        def draw_resistor(ax, x, y, width=0.4, height=0.1, vertical=False):
            if not vertical:
                xx = np.linspace(x, x + width, 8)
                yy = np.array([y, y + height, y - height, y + height, y - height, y + height, y - height, y])
                ax.plot(xx, yy, 'k-', linewidth=1.5)
            else:
                yy = np.linspace(y, y + width, 8)
                xx = np.array([x, x + height, x - height, x + height, x - height, x + height, x - height, x])
                ax.plot(xx, yy, 'k-', linewidth=1.5)
        def draw_capacitor(ax, x, y, width=0.05, gap=0.05, length=0.1, vertical=False):
            if not vertical:
                ax.plot([x, x + width], [y, y], 'k-', linewidth=1.5)
                ax.plot([x + width + gap, x + width + gap + length], [y, y], 'k-', linewidth=1.5)
                ax.plot([x + width, x + width], [y - length/2, y + length/2], 'k-', linewidth=1.5)
                ax.plot([x + width + gap, x + width + gap], [y - length/2, y + length/2], 'k-', linewidth=1.5)
            else:
                ax.plot([x, x], [y, y + width], 'k-', linewidth=1.5)
                ax.plot([x, x], [y + width + gap, y + width + gap + length], 'k-', linewidth=1.5)
                ax.plot([x - length/2, x + length/2], [y + width, y + width], 'k-', linewidth=1.5)
                ax.plot([x - length/2, x + length/2], [y + width + gap, y + width + gap], 'k-', linewidth=1.5)
        ax1.plot([0.1, 0.9], [0.5, 0.5], 'k-', linewidth=1.5)
        ax1.plot([0.1, 0.1], [0.3, 0.7], 'k-', linewidth=1.5)
        ax1.plot([0.9, 0.9], [0.3, 0.7], 'k-', linewidth=1.5)
        ax1.plot([0.1, 0.3], [0.7, 0.7], 'k-', linewidth=1.5)
        ax1.plot([0.5, 0.7], [0.7, 0.7], 'k-', linewidth=1.5)
        ax1.plot([0.3, 0.5], [0.3, 0.3], 'k-', linewidth=1.5)
        ax1.plot([0.7, 0.9], [0.3, 0.3], 'k-', linewidth=1.5)
        draw_resistor(ax1, 0.3, 0.7, 0.2, 0.05)
        draw_resistor(ax1, 0.5, 0.3, 0.2, 0.05)
        draw_capacitor(ax1, 0.7, 0.7, 0.05, 0.05, 0.1)
        draw_capacitor(ax1, 0.7, 0.3, 0.05, 0.05, 0.1)
        ax1.text(0.4, 0.78, 'R₁', ha='center')
        ax1.text(0.6, 0.22, 'R₂', ha='center')
        ax1.text(0.77, 0.78, 'C₁', ha='center')
        ax1.text(0.77, 0.22, 'C₂', ha='center')
        ax1.text(0.05, 0.5, 'Vin', ha='right')
        ax1.text(0.95, 0.5, 'Vout', ha='left')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Circuit Diagram')
        ax1.set_aspect('equal')
        ax1.axis('off')

        ax2 = plt.subplot(2, 2, 2)
        freq = np.logspace(1, 5, 1000)
        fc = 1000
        gain_db = -20 * np.log10(np.sqrt(1 + (freq/fc)**2))
        phase = -np.arctan(freq/fc) * 180 / np.pi
        ax2.semilogx(freq, gain_db, 'b-', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Gain (dB)', color='b')
        ax2.set_title('Frequency Response')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.set_xlim(10, 100000)
        ax2.set_ylim(-40, 5)
        ax2_twin = ax2.twinx()
        ax2_twin.semilogx(freq, phase, 'r-', linewidth=2)
        ax2_twin.set_ylabel('Phase (degrees)', color='r')
        ax2_twin.set_ylim(-90, 0)
        ax2_twin.tick_params(axis='y', labelcolor='r')

        ax3 = plt.subplot(2, 2, 3)
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        t_sampled = np.linspace(0, 1, 20)
        signal_sampled = np.sin(2 * np.pi * 5 * t_sampled) + 0.5 * np.sin(2 * np.pi * 10 * t_sampled)
        ax3.plot(t, signal, 'b-', linewidth=1, alpha=0.7, label='Analog Signal')
        ax3.plot(t_sampled, signal_sampled, 'ro', label='Sampled Points')
        for i, (x, y) in enumerate(zip(t_sampled, signal_sampled)):
            ax3.text(x, y + 0.3, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
            ax3.vlines(x, -1.8, -1.2, colors='r', linestyles='-')
        binary_values = ['10110011', '11001010', '01011100', '00110101', '10010110']
        for i, x in enumerate(t_sampled[:5]):
            ax3.text(x, -1.5, binary_values[i], ha='center', va='center', fontsize=7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Signal Digitization')
        ax3.set_ylim(-2, 2.5)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 2, 4)
        cpu_rect = plt.Rectangle((0.35, 0.45), 0.3, 0.2, fill=True, color='lightblue', alpha=0.8, ec='blue')
        ax4.add_patch(cpu_rect)
        ax4.text(0.5, 0.55, 'CPU', ha='center', va='center', fontsize=12)
        ram_rect = plt.Rectangle((0.2, 0.75), 0.2, 0.15, fill=True, color='lightgreen', alpha=0.8, ec='green')
        ax4.add_patch(ram_rect)
        ax4.text(0.3, 0.825, 'RAM', ha='center', va='center', fontsize=10)
        rom_rect = plt.Rectangle((0.6, 0.75), 0.2, 0.15, fill=True, color='lightgreen', alpha=0.8, ec='green')
        ax4.add_patch(rom_rect)
        ax4.text(0.7, 0.825, 'ROM', ha='center', va='center', fontsize=10)
        io_rect1 = plt.Rectangle((0.2, 0.2), 0.15, 0.15, fill=True, color='lightsalmon', alpha=0.8, ec='red')
        ax4.add_patch(io_rect1)
        ax4.text(0.275, 0.275, 'GPIO', ha='center', va='center', fontsize=8)
        io_rect2 = plt.Rectangle((0.425, 0.2), 0.15, 0.15, fill=True, color='lightsalmon', alpha=0.8, ec='red')
        ax4.add_patch(io_rect2)
        ax4.text(0.5, 0.275, 'UART', ha='center', va='center', fontsize=8)
        io_rect3 = plt.Rectangle((0.65, 0.2), 0.15, 0.15, fill=True, color='lightsalmon', alpha=0.8, ec='red')
        ax4.add_patch(io_rect3)
        ax4.text(0.725, 0.275, 'SPI/I²C', ha='center', va='center', fontsize=8)
        ax4.plot([0.5, 0.5], [0.45, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.3, 0.5], [0.75, 0.65], 'k-', linewidth=1.5)
        ax4.plot([0.7, 0.5], [0.75, 0.65], 'k-', linewidth=1.5)
        ax4.plot([0.275, 0.425], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.5, 0.5], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.575, 0.725], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.275, 0.275], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.5, 0.5], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.plot([0.725, 0.725], [0.35, 0.35], 'k-', linewidth=1.5)
        ax4.set_xlim(0.1, 0.9)
        ax4.set_ylim(0.1, 0.9)
        ax4.set_title('Microcontroller Architecture')
        ax4.axis('off')

    elif domain == 'ete' or domain.startswith('ete/'):
        # ETE visualization (remained mostly the same, can be generalized further)
        ax1 = plt.subplot(2, 2, 1)
        t = np.linspace(0, 1, 1000)
        carrier = np.sin(2 * np.pi * 20 * t)
        message = np.sin(2 * np.pi * 2 * t)
        am_signal = (1 + 0.5 * message) * carrier
        ax1.plot(t[:200], message[:200], 'g-', linewidth=1.5, label='Message Signal')
        ax1.plot(t[:200], carrier[:200], 'r-', alpha=0.3, linewidth=1, label='Carrier')
        ax1.plot(t[:200], am_signal[:200], 'b-', linewidth=1.5, label='Modulated Signal')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Amplitude Modulation')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        encoded = []
        for i in range(len(bits)):
            if i == 0:
                encoded.extend([bits[i], bits[i]])
            else:
                encoded.extend([bits[i], bits[i] ^ bits[i-1]])
        encoded = np.array(encoded)
        for i, bit in enumerate(bits):
            color = 'lightblue' if bit == 0 else 'orange'
            rect = plt.Rectangle((i, 0.7), 0.8, 0.6, fill=True, color=color, ec='black')
            ax2.add_patch(rect)
            ax2.text(i + 0.4, 1, str(bit), ha='center', va='center', fontsize=12)
        for i, bit in enumerate(encoded):
            color = 'lightblue' if bit == 0 else 'orange'
            rect = plt.Rectangle((i/2, 0), 0.4, 0.5, fill=True, color=color, ec='black')
            ax2.add_patch(rect)
            ax2.text(i/2 + 0.2, 0.25, str(bit), ha='center', va='center', fontsize=10)
        for i in range(len(bits)):
            ax2.annotate('', xy=(i/2 + 0.2, 0.6), xytext=(i + 0.4, 0.7), arrowprops=dict(arrowstyle='->', color='black', lw=1))
            if i > 0:
                ax2.annotate('', xy=(i/2 + 0.6, 0.6), xytext=(i - 0.6 + 0.4, 0.7), arrowprops=dict(arrowstyle='->', color='black', lw=1, ls='--'))
        ax2.text(4, 1.4, 'Original Data', ha='center', fontsize=10)
        ax2.text(2, -0.2, 'Encoded Data (Rate 1/2 Convolutional Code)', ha='center', fontsize=10)
        ax2.set_xlim(-0.5, 8.5)
        ax2.set_ylim(-0.4, 1.6)
        ax2.set_title('Channel Coding')
        ax2.axis('off')

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot([0.2, 0.2], [0.1, 0.4], 'k-', linewidth=2)
        ax3.plot([0.1, 0.3], [0.4, 0.4], 'k-', linewidth=2)
        ax3.scatter(0.8, 0.2, s=150, color='lightgray', edgecolors='black', zorder=2)
        ax3.text(0.8, 0.2, 'Rx', ha='center', va='center')
        for r in np.linspace(0.1, 0.6, 6):
            circle = plt.Circle((0.2, 0.4), r, fill=False, color='blue', alpha=0.15)
            ax3.add_patch(circle)
        ax3.fill([0.4, 0.6, 0.6, 0.4], [0, 0, 0.3, 0.3], color='gray', alpha=0.5)
        ax3.fill([0.7, 0.9, 0.9, 0.7], [0.5, 0.5, 0.8, 0.8], color='gray', alpha=0.5)
        ax3.plot([0.2, 0.8], [0.4, 0.2], 'r-', linewidth=1.5, label='Direct Path')
        ax3.plot([0.2, 0.5, 0.8], [0.4, 0.6, 0.2], 'g--', linewidth=1.5, label='Reflected Path')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Wireless Signal Propagation')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.axis('off')

        ax4 = plt.subplot(2, 2, 4)
        cloud_x, cloud_y = 0.5, 0.8
        router_x, router_y = 0.5, 0.4
        devices = [(0.2, 0.2, 'Smartphone'), (0.4, 0.2, 'Laptop'), (0.6, 0.2, 'IoT Device'), (0.8, 0.2, 'Smart TV')]
        device_x = [x for x, y, name in devices]
        device_y = [y for x, y, name in devices]
        device_labels = [name for x, y, name in devices]
        ax4.scatter(device_x, device_y, s=200, color='lightyellow', edgecolors='black')
        for i, label in enumerate(device_labels):
            ax4.text(device_x[i], device_y[i] - 0.08, label, ha='center', va='center', fontsize=7)
        ax4.annotate('Internet Cloud', xy=(cloud_x, cloud_y), xycoords='data', xytext=(0, 30), textcoords='offset points',
                    ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        ax4.annotate('Router', xy=(router_x, router_y), xycoords='data', xytext=(0, -30), textcoords='offset points',
                    ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))
        for x, y, name in devices:
             ax4.plot([x, router_x], [y, router_y], 'k-', linewidth=1)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Network Architecture (Scatter Plot Layout)')
        ax4.axis('off')


    else: # More Generalized Default Visualization
        # Generic scientific/technical visualization for any other or unknown domain
        ax1 = plt.subplot(2, 2, 1)
        x = np.linspace(0, 10, 100)
        y = 0.5 * x + np.sin(x) + 5
        error = 0.5 + 0.5 * np.random.random(len(x))
        ax1.plot(x, y, 'b-', linewidth=2, label='Measured Data')
        ax1.fill_between(x, y - error, y + error, color='blue', alpha=0.2, label='Uncertainty')
        fit_params = np.polyfit(x, y, 1)
        fit_line = np.poly1d(fit_params)
        ax1.plot(x, fit_line(x), 'r--', linewidth=1.5, label=f'Linear Fit (y = {fit_params[0]:.2f}x + {fit_params[1]:.2f})')
        ax1.set_xlabel('Independent Variable')
        ax1.set_ylabel('Dependent Variable')
        ax1.set_title('Data with Trend Analysis')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        categories = ['Category X', 'Category Y', 'Category Z', 'Category W']
        values = np.array([30, 25, 20, 25])
        ax2.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set2(np.arange(len(categories))))
        ax2.axis('equal')
        ax2.set_title('Category Distribution (Pie Chart)')

        ax3 = plt.subplot(2, 2, 3)
        n_points = 50
        centers = [(2, 2), (5, 7), (8, 3)]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        for i, (cx, cy) in enumerate(centers):
            cluster_x = cx + np.random.randn(n_points) * 1.0
            cluster_y = cy + np.random.randn(n_points) * 1.0
            ax3.scatter(cluster_x, cluster_y, c=colors[i], label=f'Cluster {i+1}', alpha=0.7, edgecolor='white', linewidth=0.5, s=40)
            ax3.scatter(cx, cy, c=colors[i], marker='x', s=100, linewidth=2)
        ax3.set_xlabel('Feature 1')
        ax3.set_ylabel('Feature 2')
        ax3.set_title('Multivariate Clustering')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 2, 4)
        matrix_size = 8
        correlation_matrix = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    distance = abs(i - j)
                    correlation_matrix[i, j] = np.exp(-distance / 2.0)
        noise = np.random.normal(0, 0.1, correlation_matrix.shape)
        correlation_matrix += noise
        correlation_matrix = np.clip(correlation_matrix, -1, 1)
        labels = [f'Var {i+1}' for i in range(matrix_size)]
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black', fontsize=6)
        ax4.set_xticks(np.arange(len(labels)))
        ax4.set_yticks(np.arange(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
        ax4.set_yticklabels(labels)
        plt.colorbar(im, ax=ax4, label='Correlation')
        ax4.set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.suptitle(f'Analysis for {domain.replace("_", " ").title()} Research', y=0.98, fontsize=14)

    if extracted_data and any(len(v) > 0 for v in extracted_data.values()):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        data_counts = {k: len(v) for k, v in extracted_data.items() if v and k != 'all_numbers'} # Exclude 'all_numbers' count
        if data_counts:
            labels = data_counts.keys()
            sizes = data_counts.values()
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Available Data Types (Pie Chart)')
        else:
            plt.text(0.5, 0.5, 'No numerical data extracted from paper', ha='center', va='center', fontsize=12)
            plt.axis('off')

        if extracted_data['percentages']:
            plt.subplot(2, 2, 2)
            percentages = np.array(extracted_data['percentages'])
            labels = [f'Value {i+1}' for i in range(len(percentages))]
            plt.bar(labels, percentages, color=plt.cm.tab10(np.arange(len(percentages))))
            plt.ylabel('Percentage (%)')
            plt.title('Extracted Percentage Values (Vertical Bar Chart)')
            for i, v in enumerate(percentages):
                plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

        if extracted_data['measurements']:
            plt.subplot(2, 2, 3)
            measurements = np.array(extracted_data['measurements'])
            values = measurements[:, 0]
            errors = measurements[:, 1]
            labels = [f'Measurement {i+1}' for i in range(len(measurements))]
            x_pos = np.arange(len(measurements))
            plt.errorbar(x_pos, values, yerr=errors, fmt='o', capsize=5, linestyle='-', colors=plt.cm.Pastel1(np.arange(len(measurements))))
            plt.xticks(x_pos, labels, rotation=45, ha='right')
            plt.ylabel('Value')
            plt.title('Measurements with Error Bars (Scatter Plot)')
            plt.grid(True, axis='y', alpha=0.3)

        if extracted_data['table_rows'] or extracted_data['number_sequences'] or extracted_data['value_ranges'] or extracted_data['ratios'] or extracted_data['all_numbers']: # Include all data types
            plt.subplot(2, 2, 4)
            data_to_plot = extracted_data['table_rows'] + extracted_data['number_sequences'] + extracted_data['value_ranges'] + extracted_data['ratios'] + [extracted_data['all_numbers']] # Combine all numerical data
            data_to_plot = [d for d in data_to_plot if d] # Filter out empty lists

            if data_to_plot:
                for i, row in enumerate(data_to_plot):
                    if isinstance(row, tuple) and len(row) == 2: # Handle value ranges and ratios as tuples
                        y_values = list(row)
                        x_values = [i*2, i*2 + 1] # Separate x positions for start and end of range/ratio
                        plt.plot(x_values, y_values, marker='o', linestyle='-', label=f'Range/Ratio {i+1}') # Plot as line
                    elif isinstance(row, list): # Handle lists (table rows, number sequences, all_numbers)
                        plt.plot(row, marker='o', linestyle='-', label=f'Series {i+1}')
                    else: # Fallback for other types (should not happen ideally)
                        plt.plot([row], marker='o', linestyle='-', label=f'Value {i+1}') # Plot single values

                plt.xlabel('Data Point Index')
                plt.ylabel('Value')
                plt.title('Combined Numerical Data (Line Plot)')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
            else:
                plt.text(0.5, 0.5, 'No sequence or numerical data available for detailed plot', ha='center', va='center')
                plt.axis('off')

        plt.tight_layout()
        plt.suptitle('Visualizations Using Extracted Data', y=0.98, fontsize=14)


def generate_visualization(paper_text: str, paper_images: List[Dict[str, Any]] = None) -> list:
    """
    Generate visualizations based on the research paper text and images.
    Uses LLM to create visualization code dynamically for any domain.

    Args:
        paper_text (str): Text from the research paper
        paper_images (List[Dict]): List of extracted images with metadata

    Returns:
        list: List of base64 encoded visualizations
    """
    # Detect the domain of the paper
    domain = detect_domain(paper_text)
    
    # Enhanced Prompt for multimodal input
    prompt_template = PromptTemplate.from_template("""
You are an expert visualization designer for research papers. Your goal is to create great visualizations that are insightful, visually stunning, and effectively communicate the key findings of a research paper.

Analyze the following research paper content and create Python code using matplotlib and numpy to generate compelling visualizations.

Research Paper Domain: {domain}. Consider this domain as context, but focus on visualizing the core concepts and data relationships described in the paper itself.

Research Paper Text:
{paper_text}

{images_context}

**Visualization Goals:**
***IMPORTANT:** make sure that visualizations have difference in values between the features chosen, dont fabircate new data, only use the data provided in the paper***
1. **Clarity and Insight:** The visualizations should clearly illustrate the most important findings, relationships, or concepts from the paper. Aim for visualizations that provide immediate insight and understanding.
2. **Visual Appeal ("Goated" Graphs):** Create visually engaging and aesthetically pleasing graphs. Use appropriate colors, styles, and layouts to maximize impact and memorability.
3. **Relevance:** Ensure the visualizations are directly relevant to the research paper's content. Focus on visualizing the core contribution, methodology, or key results.
4. **Diversity of Visualization Types:**  Use a variety of advanced and effective visualization types beyond basic charts. Consider:
    * **Scatter plots, Line plots, Bar charts (vertical preferred), Pie charts, Heatmaps, Area charts, Network graphs, Distribution plots (histograms, box plots, violin plots), 2D/3D visualizations where appropriate.**
    * **Creative and novel visualization approaches that best suit the data and message.**
5. **Data Representation:** If explicit numerical data or statistics are provided in the paper text, prioritize visualizing this actual data. If specific data points are not available, visualizations should conceptually represent the relationships, processes, or structures described in the text. **Strictly avoid inventing specific data values or statistics not mentioned or implied by the paper content.** Focus on accurately representing the *patterns*, *trends*, or *qualitative relationships* described.
6. **Multi-Panel Figures:** Create figures with multiple subplots to present different aspects of the research in a cohesive and comprehensive manner. Use `matplotlib.gridspec.GridSpec` for advanced layouts.
7. **Informative and Beautiful:** Include clear titles, labels, legends, annotations, and colorbars to make the visualizations self-explanatory and publication-quality.
8. **Code Quality:** Write clean, efficient, and well-commented Python code that is easily understandable and reproducible.
9. **Strict Data Adherence:** **CRITICALLY IMPORTANT: Do NOT invent or fabricate any numerical data, statistics, or specific values that are not explicitly present or directly inferable from the provided paper text. All quantitative aspects of the visualization MUST be grounded in the provided information. If the paper describes a concept qualitatively, represent it qualitatively or conceptually without inventing numbers.**

**Specific Instructions for Code Generation:**
* **Include ALL necessary import statements** at the beginning of your code.
* **Do NOT use external data files or libraries beyond common scientific Python libraries (matplotlib, numpy, scipy, pandas, scikit-learn).** Assume these are available in the execution environment.
* **For styling, use only valid matplotlib styles.** Avoid deprecated styles like 'seaborn-poster'. Use plt.style.use('seaborn-v0_8') or plt.style.use('ggplot') or simply omit style settings.
* **If you need seaborn-like aesthetics, use matplotlib's built-in styling or configure matplotlib settings directly.**
* **If feature values are very similar, choose visualization types that effectively highlight subtle differences or other data characteristics (e.g., heatmaps, scatter plots, line plots).**
* **Focus on creating visualizations that tell a story and enhance the reader's understanding of the research.**
* **IMPORTANT: Do NOT include `plt.savefig()` or `plt.close()` in your generated code. The execution environment will handle saving and closing figures.**


**Output:**
Only provide the Python code block. Do not include any introductory text, explanations, or markdown formatting outside of the code block.

```python
# Your Python code here
```
""")
    chain = prompt_template | llm | StrOutputParser()

    # Prepare image context for prompt
    images_context = ""
    if paper_images and len(paper_images) > 0:
        images_context = "The paper contains these key images:\n"
        for i, img in enumerate(paper_images[:3]):  # Limit to first 3 images
            images_context += f"- Image {i+1}: {img.get('format', 'unknown')} format, {img.get('width', 0)}x{img.get('height', 0)} pixels\n"
        images_context += "\nConsider incorporating insights from these images into your visualizations."

    visualization_code = chain.invoke({
        "paper_text": paper_text[:50000],
        "domain": domain,
        "images_context": images_context
    })
    print("Successfully generated visualization code from LLM with enhanced prompt")
    print("\\n--- LLM Generated Code START ---")
    print(visualization_code)
    print("--- LLM Generated Code END ---\\n")
    
    # Execute the code to generate visualizations
    images = safe_execute_plotting_code(visualization_code)
    
    return images


