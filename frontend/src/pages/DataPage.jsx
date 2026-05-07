import React, { useState, useEffect } from 'react';
import { 
    Database, Zap, FileJson, CheckCircle, AlertCircle, 
    Loader2, BarChart3, TrendingUp, PieChart as PieIcon, 
    Layers, Download, Activity, Globe
} from 'lucide-react';
import axios from 'axios';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, 
    Tooltip, ResponsiveContainer, Cell, AreaChart, 
    Area, PieChart, Pie 
} from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

const DataPage = () => {
    const [status, setStatus] = useState('idle'); // idle, loading, success, error
    const [result, setResult] = useState(null);
    const [viewMode, setViewMode] = useState('full'); // full, diverse
    
    // Progress state
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(null);
    const [error, setError] = useState(null);

    // Fetch initial stats
    useEffect(() => {
        fetchStats('full');
    }, []);

    // Polling for progress during long tasks
    useEffect(() => {
        let interval;
        if (isProcessing) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${import.meta.env.VITE_API_URL}/data/progress`);
                    setProgress(res.data);
                    if (res.data.status === 'idle') {
                        setIsProcessing(false);
                        fetchStats(viewMode); // Refresh results when done
                    }
                } catch (e) {
                    console.error("Progress polling error", e);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isProcessing, viewMode]);

    const fetchStats = async (mode) => {
        setStatus('loading');
        try {
            const url = `${import.meta.env.VITE_API_URL}/data/stats?mode=${mode}&_t=${Date.now()}`;
            const response = await axios.get(url);
            setResult(response.data);
            setStatus('success');
            setViewMode(mode);
        } catch (error) {
            console.error('API Failed:', error);
            setStatus('error');
            setError("Failed to load statistics");
        }
    };

    const handleAction = async (action) => {
        setIsProcessing(true);
        setError(null);
        try {
            if (action === 'generate') {
                await axios.post(`${import.meta.env.VITE_API_URL}/data/generate-ground-truth`);
            } else if (action === 'extract') {
                await axios.post(`${import.meta.env.VITE_API_URL}/data/select-diverse-gt`);
                setViewMode('diverse');
            }
        } catch (err) {
            setIsProcessing(false);
            setError(err.response?.data?.detail || "Action failed");
        }
    };

    const handleDownload = (filename) => {
        const baseUrl = import.meta.env.VITE_API_URL.replace('/api', '');
        const url = `${baseUrl}/static/ground_truth/${filename}`;
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="min-h-screen bg-[#020617] text-slate-200 p-8 animate-in fade-in duration-700">
            <div className="max-w-7xl mx-auto space-y-10">
                
                {/* Header Section */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="space-y-1">
                        <h1 className="text-4xl font-black text-white tracking-tight flex items-center gap-3">
                            <Database className="w-10 h-10 text-blue-500" />
                            DATA <span className="text-blue-500">MANAGEMENT</span>
                        </h1>
                        <p className="text-slate-400 font-medium">Dataset optimization, Ground Truth generation and cluster analysis</p>
                    </div>
                    
                    <div className="flex items-center gap-2 bg-slate-900/50 p-1.5 rounded-2xl border border-slate-800 backdrop-blur-xl">
                        <button 
                            onClick={() => fetchStats('full')}
                            disabled={isProcessing}
                            className={`px-6 py-2.5 rounded-xl font-bold transition-all text-xs tracking-widest ${viewMode === 'full' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            FULL DATASET
                        </button>
                        <button 
                            onClick={() => fetchStats('diverse')}
                            disabled={isProcessing}
                            className={`px-6 py-2.5 rounded-xl font-bold transition-all text-xs tracking-widest ${viewMode === 'diverse' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/20' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            DIVERSE SET (50)
                        </button>
                    </div>
                </div>

                {/* Progress Bar & Status */}
                {isProcessing && progress && (
                    <div className="bg-slate-900/80 backdrop-blur-md border border-blue-500/30 p-8 rounded-3xl animate-in fade-in zoom-in duration-500 shadow-2xl">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse shadow-[0_0_12px_rgba(59,130,246,0.5)]" />
                                <span className="text-sm font-bold text-blue-400 uppercase tracking-widest">{progress.message}</span>
                            </div>
                            <span className="text-sm font-black text-white font-mono">{Math.round((progress.current / progress.total) * 100)}%</span>
                        </div>
                        <div className="h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700/50 p-0.5">
                            <div 
                                className="h-full bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 rounded-full transition-all duration-700 ease-out"
                                style={{ width: `${(progress.current / progress.total) * 100}%` }}
                            />
                        </div>
                        <div className="mt-3 flex justify-between text-[10px] text-slate-500 font-bold uppercase tracking-tighter">
                            <span>Initializing Engine</span>
                            <span>{progress.current} / {progress.total} Units Processed</span>
                            <span>Finalizing Assets</span>
                        </div>
                    </div>
                )}

                {/* Action Buttons Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <button 
                        onClick={() => handleAction('generate')}
                        disabled={isProcessing}
                        className="group relative px-6 py-6 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 disabled:opacity-50 text-white rounded-3xl font-black transition-all hover:scale-[1.02] active:scale-95 shadow-xl shadow-blue-900/20 flex flex-col items-center gap-2 overflow-hidden"
                    >
                        <Zap className={`w-8 h-8 ${isProcessing ? 'animate-spin' : 'group-hover:scale-110 transition-transform'}`} />
                        <span className="text-xs uppercase tracking-widest">Generate Full GT</span>
                    </button>
                    
                    <button 
                        onClick={() => handleAction('extract')}
                        disabled={isProcessing}
                        className="group relative px-6 py-6 bg-gradient-to-br from-indigo-600 to-purple-700 hover:from-indigo-500 hover:to-purple-600 disabled:opacity-50 text-white rounded-3xl font-black transition-all hover:scale-[1.02] active:scale-95 shadow-xl shadow-indigo-900/20 flex flex-col items-center gap-2 overflow-hidden"
                    >
                        <Layers className={`w-8 h-8 ${isProcessing ? 'animate-pulse' : 'group-hover:scale-110 transition-transform'}`} />
                        <span className="text-xs uppercase tracking-widest">Extract 50 Clusters</span>
                    </button>

                    <button 
                        onClick={() => handleDownload('ground_truth.json')}
                        className="px-6 py-6 bg-slate-900/40 hover:bg-slate-800/60 text-slate-300 rounded-3xl font-black border border-slate-800 transition-all flex flex-col items-center gap-2 hover:border-slate-600"
                    >
                        <Download className="w-8 h-8 text-slate-500" />
                        <span className="text-xs uppercase tracking-widest">Download GT Full</span>
                    </button>

                    <button 
                        onClick={() => handleDownload('ground_truth_2.json')}
                        className="px-6 py-6 bg-slate-900/40 hover:bg-slate-800/60 text-slate-300 rounded-3xl font-black border border-slate-800 transition-all flex flex-col items-center gap-2 hover:border-slate-600"
                    >
                        <Download className="w-8 h-8 text-slate-500" />
                        <span className="text-xs uppercase tracking-widest">Download GT Diverse</span>
                    </button>
                </div>

                {/* Main Content Area */}
                {status === 'loading' && !isProcessing ? (
                    <div className="h-96 flex flex-col items-center justify-center gap-4 bg-slate-900/20 rounded-3xl border border-slate-800/50">
                        <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
                        <p className="text-slate-500 font-bold uppercase tracking-widest text-sm">Synchronizing Data Streams...</p>
                    </div>
                ) : status === 'error' ? (
                    <div className="h-96 flex flex-col items-center justify-center gap-6 bg-red-500/5 rounded-3xl border border-red-500/20">
                        <AlertCircle className="w-16 h-16 text-red-500" />
                        <div className="text-center space-y-2">
                            <h3 className="text-xl font-bold text-white uppercase">Sync Failed</h3>
                            <p className="text-slate-500 max-w-sm">{error}</p>
                        </div>
                        <button onClick={() => fetchStats(viewMode)} className="px-8 py-3 bg-red-600 text-white rounded-2xl font-bold hover:bg-red-500 transition-all">Retry Synchronization</button>
                    </div>
                ) : result && (
                    <div className="space-y-10 animate-in slide-in-from-bottom-6 duration-1000">
                        
                        {/* Analytics Banner */}
                        {result.analytics && (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="bg-gradient-to-br from-blue-900/20 to-blue-800/5 border border-blue-500/20 p-6 rounded-3xl flex items-center gap-5">
                                    <div className="w-14 h-14 bg-blue-500/10 rounded-2xl flex items-center justify-center border border-blue-500/20">
                                        <Globe className="w-7 h-7 text-blue-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-blue-500/80 uppercase font-black tracking-widest">Candidate Pool</p>
                                        <p className="text-3xl font-black text-white font-mono leading-none">{result.analytics.total_candidates}</p>
                                    </div>
                                </div>
                                <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/5 border border-emerald-500/20 p-6 rounded-3xl flex items-center gap-5">
                                    <div className="w-14 h-14 bg-emerald-500/10 rounded-2xl flex items-center justify-center border border-emerald-500/20">
                                        <CheckCircle className="w-7 h-7 text-emerald-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-emerald-500/80 uppercase font-black tracking-widest">Perfectly Unique</p>
                                        <p className="text-3xl font-black text-white font-mono leading-none">{result.analytics.perfectly_unique}/50</p>
                                    </div>
                                </div>
                                <div className="bg-gradient-to-br from-amber-900/20 to-amber-800/5 border border-amber-500/20 p-6 rounded-3xl flex items-center gap-5">
                                    <div className="w-14 h-14 bg-amber-500/10 rounded-2xl flex items-center justify-center border border-amber-500/20">
                                        <Activity className="w-7 h-7 text-amber-400" />
                                    </div>
                                    <div>
                                        <p className="text-[10px] text-amber-500/80 uppercase font-black tracking-widest">Diversity Score</p>
                                        <p className="text-3xl font-black text-white font-mono leading-none">{((result.analytics.perfectly_unique / 50) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Visual Charts Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Similarity Distribution */}
                            <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800 p-8 rounded-[2rem] space-y-6">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <TrendingUp className="w-5 h-5 text-blue-500" />
                                        <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest">Similarity Distribution</h3>
                                    </div>
                                    <div className="px-3 py-1 bg-blue-500/10 rounded-lg border border-blue-500/20">
                                        <span className="text-xs font-mono text-blue-400">AVG: {(result.avg_overall_sim * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                                <div className="h-[250px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={result.stats}>
                                            <defs>
                                                <linearGradient id="colorSim" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                            <XAxis dataKey="range" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                                            <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                                            <Tooltip 
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.5)' }}
                                                itemStyle={{ color: '#3b82f6', fontWeight: 'bold' }}
                                            />
                                            <Area type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={4} fillOpacity={1} fill="url(#colorSim)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Dataset Coverage */}
                            <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800 p-8 rounded-[2rem] space-y-6">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <PieIcon className="w-5 h-5 text-emerald-500" />
                                        <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest">Dataset Coverage</h3>
                                    </div>
                                    <div className="px-3 py-1 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                                        <span className="text-xs font-mono text-emerald-400">{result.coverage.percentage.toFixed(1)}%</span>
                                    </div>
                                </div>
                                <div className="h-[250px] w-full flex items-center justify-center relative">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={result.coverage.overlap}
                                                innerRadius={75}
                                                outerRadius={100}
                                                paddingAngle={8}
                                                dataKey="value"
                                                stroke="none"
                                            >
                                                {result.coverage.overlap.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip 
                                                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px' }}
                                            />
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                                        <p className="text-3xl font-black text-white leading-none">{result.coverage.unique_images}</p>
                                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Unique</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Top Hubs Bar Chart */}
                        <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800 p-10 rounded-[2rem] space-y-8">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <BarChart3 className="w-6 h-6 text-amber-500" />
                                    <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest">Top Hub Images (Frequency Map)</h3>
                                </div>
                                <div className="flex gap-4">
                                    <div className="text-right">
                                        <p className="text-[10px] text-slate-500 font-bold uppercase">Total Samples</p>
                                        <p className="text-sm font-mono text-white">{result.coverage.total_images}</p>
                                    </div>
                                </div>
                            </div>
                            <div className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={result.hub_images} layout="vertical" margin={{ left: 120, right: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                        <XAxis type="number" hide />
                                        <YAxis 
                                            dataKey="name" 
                                            type="category" 
                                            stroke="#94a3b8" 
                                            fontSize={11} 
                                            width={110} 
                                            tickLine={false}
                                            axisLine={false}
                                            tickFormatter={(val) => val.length > 20 ? val.substring(0, 17) + '...' : val}
                                        />
                                        <Tooltip 
                                            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px' }}
                                        />
                                        <Bar dataKey="count" fill="#f59e0b" radius={[0, 8, 8, 0]} barSize={30}>
                                            {result.hub_images.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.08} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Footer Status */}
                        <div className="flex items-center justify-center gap-4 py-6 border-t border-slate-800/50">
                            <div className="flex items-center gap-2">
                                <Activity className="w-4 h-4 text-slate-500" />
                                <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Engine Status: Optimal</span>
                            </div>
                            <div className="w-1 h-1 bg-slate-700 rounded-full" />
                            <div className="flex items-center gap-2">
                                <FileJson className="w-4 h-4 text-slate-500" />
                                <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Source: {result.source}</span>
                            </div>
                            <div className="w-1 h-1 bg-slate-700 rounded-full" />
                            <div className="flex items-center gap-2">
                                <Loader2 className="w-4 h-4 text-slate-500" />
                                <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Last Sync: {new Date(result.timestamp * 1000).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Empty State */}
                {!result && status === 'idle' && (
                    <div className="h-[60vh] flex flex-col items-center justify-center text-center space-y-6 bg-slate-900/10 rounded-[3rem] border border-slate-800/30">
                        <div className="w-24 h-24 bg-slate-900/50 rounded-full flex items-center justify-center border border-slate-800 shadow-inner">
                            <Database className="w-10 h-10 text-slate-700" />
                        </div>
                        <div className="space-y-2">
                            <h3 className="text-2xl font-black text-white uppercase tracking-tight">No Data Stream Detected</h3>
                            <p className="text-slate-500 max-w-sm mx-auto font-medium">Initialize the Ground Truth generation engine to start analyzing your dataset perceptual structure.</p>
                        </div>
                        <button onClick={generateGT} className="px-10 py-4 bg-blue-600 text-white rounded-2xl font-black hover:bg-blue-500 transition-all shadow-xl shadow-blue-900/20 active:scale-95 uppercase tracking-widest text-xs">Initialize Engine</button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default DataPage;
