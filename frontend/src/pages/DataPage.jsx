import React, { useState, useEffect } from 'react';
import { Database, Zap, FileJson, CheckCircle, AlertCircle, Loader2, BarChart3, TrendingUp, PieChart as PieIcon, Layers } from 'lucide-react';
import axios from 'axios';
import { 
    BarChart, 
    Bar, 
    XAxis, 
    YAxis, 
    CartesianGrid, 
    Tooltip, 
    ResponsiveContainer, 
    Cell,
    AreaChart,
    Area,
    PieChart,
    Pie
} from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

const DataPage = () => {
    const [status, setStatus] = useState('idle'); // idle, loading, success, error
    const [result, setResult] = useState(null);

    const [diverseStatus, setDiverseStatus] = useState('idle');
    const [diverseResult, setDiverseResult] = useState(null);
    const [viewMode, setViewMode] = useState('full'); // full, diverse

    // Auto-fetch stats on mount
    useEffect(() => {
        fetchStats('full');
    }, []);

    const fetchStats = async (mode) => {
        console.log(`🚀 FETCHING STATS FOR: ${mode} at ${new Date().toLocaleTimeString()}`);
        setStatus('loading');
        try {
            const url = `http://localhost:8000/api/data/stats?mode=${mode}&_t=${Date.now()}`;
            console.log(`📡 Calling API: ${url}`);
            const response = await axios.get(url);
            console.log('✅ API RESPONSE:', response.data);
            setResult(response.data);
            setStatus('success');
        } catch (error) {
            console.error('❌ API FAILED:', error);
            setStatus('error');
        }
    };

    const generateGroundTruth = async () => {
        setStatus('loading');
        try {
            const response = await axios.post('http://localhost:8000/api/data/generate-ground-truth');
            setResult(response.data);
            setStatus('success');
            setViewMode('full');
        } catch (error) {
            console.error('Failed to generate ground truth:', error);
            setStatus('error');
        }
    };

    const selectDiverseGT = async () => {
        setDiverseStatus('loading');
        try {
            const response = await axios.post('http://localhost:8000/api/data/select-diverse-gt');
            setDiverseResult(response.data);
            setDiverseStatus('success');
            setResult(response.data);
            setStatus('success');
            setViewMode('diverse');
        } catch (error) {
            console.error('Failed to select diverse GT:', error);
            setDiverseStatus('error');
        }
    };

    const handleToggleView = (mode) => {
        setViewMode(mode);
        fetchStats(mode);
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 pb-20">
            {/* Header section */}
            <div className="flex flex-col gap-2">
                <h1 className="text-4xl font-black text-white tracking-tight flex items-center gap-4">
                    <Database className="w-10 h-10 text-primary-500" />
                    Data Management
                </h1>
                <p className="text-slate-400 text-lg max-w-2xl">
                    Manage and evaluate your dataset perceptual structure.
                </p>
            </div>

            {/* Action Cards Row - 2 Columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Generation Card */}
                <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-8 rounded-3xl shadow-2xl relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Zap className="w-24 h-24 text-primary-500 rotate-12" />
                    </div>
                    <div className="relative z-10 w-full">
                        <button
                            onClick={generateGroundTruth}
                            disabled={status === 'loading'}
                            className={`w-full py-6 rounded-2xl font-bold uppercase tracking-widest text-sm flex items-center justify-center gap-3 transition-all duration-300 ${
                                status === 'loading'
                                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                    : 'bg-primary-600 hover:bg-primary-500 text-white shadow-xl shadow-primary-600/20 active:scale-[0.98]'
                            }`}
                        >
                            {status === 'loading' ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5" />
                                    Start Full Generation
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Selection Card */}
                <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-8 rounded-3xl shadow-2xl relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                        <Layers className="w-24 h-24 text-emerald-500 -rotate-12" />
                    </div>
                    <div className="relative z-10 w-full space-y-4">
                        <button
                            onClick={selectDiverseGT}
                            disabled={diverseStatus === 'loading'}
                            className={`w-full py-6 rounded-2xl font-bold uppercase tracking-widest text-sm flex items-center justify-center gap-3 transition-all duration-300 ${
                                diverseStatus === 'loading'
                                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-xl shadow-emerald-600/20 active:scale-[0.98]'
                            }`}
                        >
                            {diverseStatus === 'loading' ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Selecting...
                                </>
                            ) : (
                                <>
                                    <CheckCircle className="w-5 h-5" />
                                    Extract 50 Clusters
                                </>
                            )}
                        </button>
                        {diverseStatus === 'success' && (
                            <div className="p-3 bg-emerald-500/10 rounded-xl border border-emerald-500/20 animate-in fade-in zoom-in duration-300 flex items-center justify-between">
                                <span className="text-[10px] text-emerald-500 uppercase font-black tracking-widest">Diverse Set Ready</span>
                                <span className="text-xs text-slate-400 font-mono">Unique: {diverseResult?.coverage?.unique_images}</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Charts & Stats Area - Full Width */}
            <div className="space-y-8">
                {status === 'success' && (
                    <div className="flex items-center justify-center gap-1 bg-slate-900/50 p-1 rounded-2xl border border-slate-800 w-fit mx-auto">
                        <button
                            onClick={() => handleToggleView('full')}
                            className={`px-6 py-2 rounded-xl text-xs font-bold transition-all duration-300 ${
                                viewMode === 'full' 
                                ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/20' 
                                : 'text-slate-500 hover:text-slate-300'
                            }`}
                        >
                            FULL DATASET
                        </button>
                        <button
                            onClick={() => handleToggleView('diverse')}
                            className={`px-6 py-2 rounded-xl text-xs font-bold transition-all duration-300 ${
                                viewMode === 'diverse' 
                                ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20' 
                                : 'text-slate-500 hover:text-slate-300'
                            }`}
                        >
                            DIVERSE TEST SET
                        </button>
                    </div>
                )}

                {status === 'idle' && (
                    <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-12 rounded-3xl shadow-2xl flex flex-col justify-center items-center text-center space-y-6">
                        <div className="w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center border border-slate-700">
                            <Database className="w-10 h-10 text-slate-600" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-xl font-bold text-white">No Statistics Loaded</h3>
                            <p className="text-slate-500 max-w-xs">Run the DreamSim generation to analyze your dataset perceptual structure.</p>
                        </div>
                    </div>
                )}

                {status === 'error' && (
                    <div className="bg-red-500/10 backdrop-blur-xl border border-red-500/20 p-12 rounded-3xl shadow-2xl flex flex-col justify-center items-center text-center space-y-6">
                        <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center border border-red-500/30">
                            <AlertCircle className="w-10 h-10 text-red-500" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-xl font-bold text-white">Failed to load {viewMode} statistics</h3>
                            <p className="text-slate-500 max-w-xs">Make sure the ground truth file exists and is valid.</p>
                        </div>
                        <button 
                            onClick={() => fetchStats(viewMode)}
                            className="px-8 py-3 bg-red-600 hover:bg-red-500 text-white rounded-2xl font-bold transition-all"
                        >
                            Retry
                        </button>
                    </div>
                )}

                {status === 'loading' && (
                    <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-12 rounded-3xl shadow-2xl flex flex-col justify-center items-center text-center space-y-8">
                        <div className="w-24 h-24 relative">
                            <div className="absolute inset-0 border-4 border-primary-500/20 rounded-full" />
                            <div className="absolute inset-0 border-4 border-primary-500 border-t-transparent rounded-full animate-spin" />
                            <div className="absolute inset-0 flex items-center justify-center">
                                <Layers className="w-10 h-10 text-primary-500 animate-pulse" />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-xl font-bold text-white">Analyzing Relationships</h3>
                            <p className="text-slate-400">This involves N x TopK comparisons and frequency mapping.</p>
                        </div>
                    </div>
                )}

                {status === 'success' && (
                    <div className="space-y-8 animate-in zoom-in-95 duration-500">
                        {/* Distribution & Coverage Row */}
                        <div className="flex items-center justify-between px-2">
                            <div className="flex items-center gap-3">
                                <FileJson className="w-4 h-4 text-slate-500" />
                                <span className="text-lg font-bold text-red-500 uppercase tracking-tighter">
                                    DEBUG SOURCE: {result?.source} • TIME: {new Date(result?.timestamp * 1000).toLocaleTimeString()}
                                </span>
                            </div>
                        </div>

                        {/* Analytics Cards (Diverse Set Only) */}
                        {result?.analytics && (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in slide-in-from-top-4 duration-700">
                                <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-2xl flex items-center gap-4">
                                    <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center border border-blue-500/20">
                                        <Layers className="w-6 h-6 text-blue-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500 uppercase font-bold">Candidate Pool</p>
                                        <p className="text-xl font-mono text-white font-bold">{result.analytics.total_candidates}</p>
                                    </div>
                                </div>
                                <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-2xl flex items-center gap-4">
                                    <div className="w-12 h-12 bg-emerald-500/10 rounded-xl flex items-center justify-center border border-emerald-500/20">
                                        <CheckCircle className="w-6 h-6 text-emerald-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500 uppercase font-bold">Perfectly Unique</p>
                                        <p className="text-xl font-mono text-white font-bold">{result.analytics.perfectly_unique_selected}/50</p>
                                    </div>
                                </div>
                                <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-2xl flex items-center gap-4">
                                    <div className="w-12 h-12 bg-amber-500/10 rounded-xl flex items-center justify-center border border-amber-500/20">
                                        <Zap className="w-6 h-6 text-amber-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-slate-500 uppercase font-bold">Diversity Score</p>
                                        <p className="text-xl font-mono text-white font-bold">
                                            {((result.analytics.perfectly_unique_selected / 50) * 100).toFixed(1)}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-6 rounded-3xl">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-3">
                                        <TrendingUp className="w-5 h-5 text-primary-500" />
                                        <h3 className="font-bold text-white uppercase text-xs tracking-widest">Similarity Dist.</h3>
                                    </div>
                                    <p className="text-sm font-mono text-primary-400">Avg: {(result?.avg_overall_sim * 100).toFixed(1)}%</p>
                                </div>
                                <div className="h-[200px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={result?.stats}>
                                            <XAxis dataKey="range" hide />
                                            <Tooltip 
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                            />
                                            <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={3} />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-6 rounded-3xl">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-3">
                                        <PieIcon className="w-5 h-5 text-emerald-500" />
                                        <h3 className="font-bold text-white uppercase text-xs tracking-widest">Dataset Coverage</h3>
                                    </div>
                                    <p className="text-sm font-mono text-emerald-400">{result?.coverage?.percentage.toFixed(1)}%</p>
                                </div>
                                <div className="h-[200px] w-full flex items-center justify-center">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={result?.coverage?.overlap}
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                            >
                                                {result?.coverage?.overlap.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip 
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                            />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>

                        <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-8 rounded-3xl">
                            <div className="flex items-center gap-3 mb-8">
                                <BarChart3 className="w-5 h-5 text-amber-500" />
                                <h3 className="font-bold text-white uppercase text-xs tracking-widest">Top Hub Images (Frequency)</h3>
                            </div>
                            <div className="h-[350px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={result?.hub_images} layout="vertical" margin={{ left: 120 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                        <XAxis type="number" hide />
                                        <YAxis 
                                            dataKey="name" 
                                            type="category" 
                                            stroke="#94a3b8" 
                                            fontSize={11} 
                                            width={110} 
                                            tickFormatter={(val) => val.length > 20 ? val.substring(0, 17) + '...' : val}
                                        />
                                        <Tooltip 
                                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                        />
                                        <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} barSize={25}>
                                            {result?.hub_images.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.05} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {[
                                { label: 'Total Clusters', val: result?.count, color: 'text-primary-500' },
                                { label: 'Total Images', val: result?.coverage?.total_images, color: 'text-slate-300' },
                                { label: 'Unique Images', val: result?.coverage?.unique_images, color: 'text-emerald-500' },
                                { label: 'Max Hub Freq.', val: result?.hub_images?.[0]?.count, color: 'text-amber-500' }
                            ].map((stat, i) => (
                                <div key={i} className="bg-slate-950/50 p-4 rounded-2xl border border-slate-800">
                                    <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">{stat.label}</p>
                                    <p className={`text-xl font-bold ${stat.color}`}>{stat.val}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default DataPage;
