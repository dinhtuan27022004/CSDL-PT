import React, { useState, useRef, useEffect } from 'react';
import useImageStore from '../../store/useImageStore';
import Card from '../common/Card';
import Button from '../common/Button';
import { 
    Upload, 
    FileImage, 
    X, 
    Image as ImageIcon, 
    Search, 
    Settings2, 
    ChevronDown, 
    ChevronUp,
    Zap,
    SlidersHorizontal,
    Equal
} from 'lucide-react';

const COLOR_SPACES = ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"];
const METHODS = ["std", "interp", "gauss"];

const SEARCH_FEATURES = [
    { key: 'brightness', label: 'Brightness' },
    { key: 'contrast', label: 'Contrast' },
    { key: 'saturation', label: 'Saturation' },
    { key: 'edge_density', label: 'Edge Den.' },
    { key: 'sharpness', label: 'Sharpness' },
    
    // Traditional
    { key: 'hog', label: 'HOG' },
    { key: 'hu_moments', label: 'Hu Moments' },
    { key: 'lbp', label: 'LBP' },
    { key: 'color_moments', label: 'Moments' },
    { key: 'gabor', label: 'Gabor' },
    { key: 'ccv', label: 'CCV' },
    { key: 'zernike', label: 'Zernike' },
    { key: 'geo', label: 'Geometry' },
    { key: 'tamura', label: 'Tamura' },
    { key: 'edge_orientation', label: 'Edge Orient' },
    { key: 'glcm', label: 'GLCM' },
    { key: 'wavelet', label: 'Wavelet' },
    { key: 'correlogram', label: 'Correlogram' },
    { key: 'dominant_color', label: 'Dom Color' },

    // Advanced Traditional (Object/Spatial)
    { key: 'ehd', label: 'MPEG-7 EHD', group: 'advanced' },
    { key: 'cld', label: 'MPEG-7 CLD', group: 'advanced' },
    { key: 'spm', label: 'Spatial Pyr.', group: 'advanced' },
    { key: 'saliency', label: 'Saliency', group: 'advanced' },
    { key: 'bovw', label: 'BoVW', group: 'advanced' },

    // Semantic / Metadata
    { key: 'semantic', label: 'Semantic' },
    { key: 'category', label: 'Category' },
    { key: 'entity', label: 'Entity' },
    { key: 'dreamsim', label: 'DreamSim' },

    // Multi-Space Histograms
    ...COLOR_SPACES.flatMap(space => METHODS.map(method => ({
        key: `${space}_hist_${method}`,
        label: `${space.toUpperCase()} Hist (${method})`,
        group: space
    }))),

    // Multi-Space CDFs
    ...COLOR_SPACES.flatMap(space => METHODS.map(method => ({
        key: `${space}_cdf_${method}`,
        label: `${space.toUpperCase()} CDF (${method})`,
        group: space
    }))),

    // Joint Histograms
    ...COLOR_SPACES.filter(s => s !== 'gray').flatMap(space => METHODS.map(method => ({
        key: `joint_${space}_${method}`,
        label: `Joint ${space.toUpperCase()} (${method})`,
        group: space
    }))),

    // Cell Vectors
    ...COLOR_SPACES.map(space => ({
        key: `cell_${space}`,
        label: `Cell ${space.toUpperCase()}`,
        group: space
    }))
];

const SimilaritySearchForm = () => {
    const { searchImages, similarityLoading, fetchWeights, currentWeights } = useImageStore();
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [limit, setLimit] = useState(20);
    const [showSettings, setShowSettings] = useState(false);
    
    // Search Mode: 'optimized' | 'manual' | 'equal'
    const [mode, setMode] = useState('optimized');
    const [activeGroup, setActiveGroup] = useState('basic');
    const [manualWeights, setManualWeights] = useState(
        SEARCH_FEATURES.reduce((acc, f) => ({ ...acc, [f.key]: 1.0 }), {})
    );

    const inputRef = useRef(null);

    useEffect(() => {
        if (showSettings && !currentWeights) {
            fetchWeights();
        }
    }, [showSettings, currentWeights, fetchWeights]);

    const handleWeightChange = (key, value) => {
        setManualWeights(prev => ({ 
            ...prev, 
            [key]: parseFloat(value) || 0 
        }));
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleFiles = (newFiles) => {
        if (!newFiles || newFiles.length === 0) return;

        if (newFiles.length > 1) {
            useImageStore.getState().addToast('info', 'Similarity search supports one query image at a time. To upload multiple images to the database, please use the Import page.');
        }

        const selectedFile = newFiles[0];
        if (selectedFile && selectedFile.type.startsWith('image/')) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files) {
            handleFiles(e.dataTransfer.files);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files) {
            handleFiles(e.target.files);
        }
    };

    const removeFile = () => {
        setFile(null);
        setPreviewUrl(null);
        if (inputRef.current) {
            inputRef.current.value = '';
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return;

        const searchSettings = {
            mode,
            weights: mode === 'manual' ? manualWeights : null,
            limit
        };

        await searchImages(file, limit, searchSettings);
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-primary-500/20 flex items-center justify-center text-primary-500">
                        <Search className="w-5 h-5" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white">Similarity Search</h3>
                        <p className="text-xs text-slate-500">Find images with similar features</p>
                    </div>
                </div>
                <button 
                    onClick={() => setShowSettings(!showSettings)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-200 ${
                        showSettings 
                        ? 'bg-primary-500/20 border-primary-500/50 text-primary-400 shadow-lg shadow-primary-500/10' 
                        : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600 hover:bg-slate-750'
                    }`}
                >
                    <Settings2 className="w-4 h-4" />
                    <span className="text-sm font-semibold">Weighting System</span>
                    {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6" onDragEnter={handleDrag}>
                {showSettings && (
                    <div className="bg-slate-900/50 border border-slate-700/50 rounded-2xl p-6 space-y-6 animate-in fade-in slide-in-from-top-4 duration-300">
                        {/* Mode Selection */}
                        <div className="flex flex-col md:flex-row gap-4">
                            <button
                                type="button"
                                onClick={() => setMode('optimized')}
                                className={`flex-1 flex items-center gap-3 p-4 rounded-xl border transition-all ${
                                    mode === 'optimized'
                                    ? 'bg-primary-500/10 border-primary-500/50 text-primary-400'
                                    : 'bg-slate-800/30 border-slate-700 text-slate-500 hover:border-slate-600'
                                }`}
                            >
                                <Zap className="w-5 h-5" />
                                <div className="text-left">
                                    <p className="text-sm font-bold uppercase tracking-tight">Optimized</p>
                                    <p className="text-[10px] opacity-70">Use Bayesian best weights</p>
                                </div>
                            </button>
                            <button
                                type="button"
                                onClick={() => setMode('manual')}
                                className={`flex-1 flex items-center gap-3 p-4 rounded-xl border transition-all ${
                                    mode === 'manual'
                                    ? 'bg-primary-500/10 border-primary-500/50 text-primary-400'
                                    : 'bg-slate-800/30 border-slate-700 text-slate-500 hover:border-slate-600'
                                }`}
                            >
                                <SlidersHorizontal className="w-5 h-5" />
                                <div className="text-left">
                                    <p className="text-sm font-bold uppercase tracking-tight">Manual</p>
                                    <p className="text-[10px] opacity-70">Fine-tune weights manually</p>
                                </div>
                            </button>
                        </div>
                        
                        {/* Optimized Mode Info */}
                        {mode === 'optimized' && (
                            <div className="space-y-3 pt-2 border-t border-slate-700/30 animate-in fade-in duration-500">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                                        <Zap className="w-3 h-3 text-primary-500" /> Ground Truth Optimization
                                    </span>
                                </div>
                                <p className="text-sm text-slate-400">
                                    Ranking weights are automatically optimized based on folder-based Ground Truth to maximize retrieval accuracy.
                                </p>
                            </div>
                        )}

                        {/* Weight Inputs (Manual Mode) */}
                        {mode === 'manual' && (
                            <div className="space-y-4 pt-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Attribute Weighting Matrix</span>
                                    <div className="flex gap-4">
                                        <button 
                                            type="button" 
                                            onClick={() => setManualWeights(SEARCH_FEATURES.reduce((acc, f) => ({ ...acc, [f.key]: 1.0 }), {}))}
                                            className="text-[10px] text-slate-500 hover:text-slate-400 flex items-center gap-1 transition-colors"
                                        >
                                            <X className="w-3 h-3" /> Reset to 1.0
                                        </button>
                                        <button 
                                            type="button" 
                                            onClick={() => setManualWeights(SEARCH_FEATURES.reduce((acc, f) => ({ ...acc, [f.key]: 0.0 }), {}))}
                                            className="text-[10px] text-red-500/70 hover:text-red-500 flex items-center gap-1 transition-colors"
                                        >
                                            <X className="w-3 h-3" /> Reset to 0
                                        </button>
                                    </div>
                                </div>
                                
                                {/* Feature Groups Tabs */}
                                <div className="flex flex-wrap gap-1 border-b border-slate-700/50 pb-2">
                                    {['basic', 'advanced', 'deep', ...COLOR_SPACES].map(group => (
                                        <button
                                            key={group}
                                            type="button"
                                            onClick={() => setActiveGroup(group)}
                                            className={`px-3 py-1 rounded-t-lg text-[10px] font-black uppercase tracking-wider transition-all ${
                                                activeGroup === group 
                                                ? 'bg-primary-600 text-white shadow-lg' 
                                                : 'bg-slate-800 text-slate-500 hover:bg-slate-750'
                                            }`}
                                        >
                                            {group}
                                        </button>
                                    ))}
                                </div>

                                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                                    {SEARCH_FEATURES.filter(f => {
                                        if (activeGroup === 'basic') return !f.group && !['semantic', 'category', 'entity', 'dreamsim'].includes(f.key);
                                        if (activeGroup === 'deep') return ['semantic', 'category', 'entity', 'dreamsim'].includes(f.key);
                                        return f.group === activeGroup;
                                    }).map((feature) => (
                                        <div key={feature.key} className="flex items-center justify-between gap-2 p-2 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                                            <label className="text-[10px] font-bold text-slate-400 truncate uppercase tracking-tighter flex-1">
                                                {feature.label}
                                            </label>
                                            <input 
                                                type="number" 
                                                step="0.1"
                                                value={manualWeights[feature.key]}
                                                onChange={(e) => handleWeightChange(feature.key, e.target.value)}
                                                className="w-16 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-[10px] text-white text-right focus:outline-none focus:border-primary-500/50"
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Display current optimized weights as info when in optimized mode */}
                        {mode === 'optimized' && currentWeights && (
                            <div className="p-4 bg-primary-500/5 rounded-xl border border-primary-500/20">
                                <p className="text-xs text-primary-400/80 italic">
                                    System is using the best-performing weights found during the last evaluation run.
                                </p>
                            </div>
                        )}
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
                    <div className="md:col-span-2">
                        {!file ? (
                            <div
                                className={`relative border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 ${dragActive
                                    ? "border-primary-500 bg-primary-500/10 scale-[1.01]"
                                    : "border-slate-700 hover:border-slate-600 bg-slate-800/20"
                                    }`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                            >
                                <input
                                    ref={inputRef}
                                    type="file"
                                    accept="image/*"
                                    onChange={handleChange}
                                    className="hidden"
                                />

                                <div className="w-16 h-16 rounded-2xl bg-slate-800 mx-auto flex items-center justify-center text-slate-500 mb-4 border border-slate-700 shadow-inner">
                                    <FileImage className="w-8 h-8" />
                                </div>

                                <p className="text-lg text-slate-200 font-bold mb-1">
                                    Upload
                                </p>
                                <Button
                                    type="button"
                                    variant="secondary"
                                    onClick={onButtonClick}
                                    disabled={similarityLoading}
                                    className="px-8"
                                >
                                    Select Image
                                </Button>
                            </div>
                        ) : (
                            <div className="bg-slate-800/40 rounded-2xl p-6 flex items-center justify-between border border-slate-700 shadow-lg">
                                <div className="flex items-center gap-6">
                                    <div className="w-24 h-24 rounded-xl bg-slate-700 overflow-hidden shrink-0 border border-slate-600 shadow-2xl">
                                        <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                                    </div>
                                    <div>
                                        <h4 className="text-lg font-bold text-white">{file.name}</h4>
                                        <p className="text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB • Image Ready</p>
                                    </div>
                                </div>
                                <button
                                    type="button"
                                    onClick={removeFile}
                                    className="w-10 h-10 flex items-center justify-center hover:bg-red-500/20 rounded-xl text-slate-400 hover:text-red-400 transition-all"
                                >
                                    <X className="w-6 h-6" />
                                </button>
                            </div>
                        )}
                    </div>

                    <div className="space-y-4">
                        <div className="bg-slate-800/30 p-4 rounded-2xl border border-slate-700/50">
                            <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">
                                Results Count
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="500"
                                value={limit}
                                onChange={(e) => setLimit(parseInt(e.target.value) || 20)}
                                className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-2.5 text-white font-mono focus:outline-none focus:border-primary-500 transition-colors"
                            />
                        </div>
                        <Button
                            type="submit"
                            variant="primary"
                            className="w-full py-4 text-lg font-bold shadow-xl shadow-primary-600/20 active:scale-[0.98]"
                            disabled={similarityLoading || !file}
                        >
                            {similarityLoading ? (
                                <div className="flex items-center gap-3">
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    <span>Searching...</span>
                                </div>
                            ) : (
                                <div className="flex items-center justify-center gap-3">
                                    <Search className="w-5 h-5" />
                                    <span>Find Similar</span>
                                </div>
                            )}
                        </Button>
                    </div>
                </div>
            </form>
        </Card>
    );
};

export default SimilaritySearchForm;
