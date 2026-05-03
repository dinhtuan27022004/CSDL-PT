import { create } from 'zustand';

/**
 * Zustand Store for Image Similarity Application
 * Manages global state for images, similarity results, and UI state
 */

const API_BASE_URL = 'http://localhost:8000';

const useImageStore = create((set, get) => ({
    // Images state
    images: [],
    selectedImage: null,
    imagesLoading: false,
    imagesError: null,

    // Search and filter
    searchQuery: '',
    sortBy: 'newest', // 'newest' | 'oldest' | 'feature-ready'

    // Similarity state
    similarityResults: [],
    gtResults: [], // Added for comparison mode
    queryImageFeatures: null,
    queryImagePreviewUrl: null,
    similarityLoading: false,
    similarityError: null,
    topK: parseInt(import.meta.env.VITE_DEFAULT_TOPK) || 5,
    metric: 'cosine', // 'cosine' | 'l2'

    // Evaluation state
    evaluationData: null,
    evaluationAllData: {},
    evaluationLoading: false,
    evaluationError: null,
    isOptimizing: false,
    setIsOptimizing: (val) => set({ isOptimizing: val }),
    currentWeights: null,

    // Global sync scroll position
    syncScrollLeft: 0,
    setSyncScrollLeft: (scrollLeft) => set({ syncScrollLeft: scrollLeft }),

    // Selected result for metadata viewing
    selectedResult: null,

    /**
     * Set search query
     */
    setSearchQuery: (query) => {
        set({ searchQuery: query });
    },

    /**
     * Set sort option
     */
    setSortBy: (sortBy) => {
        set({ sortBy });
    },

    /**
     * Get filtered and sorted images
     */
    getFilteredImages: () => {
        const { images, searchQuery, sortBy } = get();

        // Filter by search
        let filtered = images;
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = images.filter(img =>
                img.file_name?.toLowerCase().includes(query) ||
                img.id?.toString().includes(query)
            );
        }

        // Sort
        const sorted = [...filtered].sort((a, b) => {
            switch (sortBy) {
                case 'oldest':
                    return new Date(a.created_at) - new Date(b.created_at);
                case 'newest':
                    return new Date(b.created_at) - new Date(a.created_at);
                case 'feature-ready':
                    return (b.brightness ? 1 : 0) - (a.brightness ? 1 : 0);
                default:
                    return 0;
            }
        });

        return sorted;
    },

    /**
     * Set similarity parameters
     */
    setTopK: (topK) => {
        set({ topK });
    },

    setMetric: (metric) => {
        set({ metric });
    },

    /**
     * Find similar images
     */
    searchImages: async (file, limit = 5, searchSettings = null) => {
        const previewUrl = URL.createObjectURL(file);
        set({ similarityLoading: true, similarityError: null, similarityResults: [], queryImageFeatures: null, queryImagePreviewUrl: previewUrl });

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('limit', limit);
            if (searchSettings) {
                formData.append('search_settings', JSON.stringify(searchSettings));
            }

            const searchUrl = `${API_BASE_URL}/api/images/search`;
            console.log('Fetching:', searchUrl);

            const response = await fetch(searchUrl, {
                method: 'POST',
                body: formData
            });


            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            const data = await response.json();

            const resultsWithPreviews = data.results.map(item => ({
                ...item,
                previewUrl: (item.url || item.file_path) ? `${API_BASE_URL}${item.url || item.file_path}` : null,
                hogPreviewUrl: item.hog_vis_path ? `${API_BASE_URL}${item.hog_vis_path}` : null,
                huPreviewUrl: item.hu_vis_path ? `${API_BASE_URL}${item.hu_vis_path}` : null,
                cellColorPreviewUrl: item.cell_color_vis_path ? `${API_BASE_URL}${item.cell_color_vis_path}` : null,
                lbpPreviewUrl: item.lbp_vis_path ? `${API_BASE_URL}${item.lbp_vis_path}` : null,
                gaborPreviewUrl: item.gabor_vis_path ? `${API_BASE_URL}${item.gabor_vis_path}` : null,
                ccvPreviewUrl: item.ccv_vis_path ? `${API_BASE_URL}${item.ccv_vis_path}` : null,
                histogramPreviewUrl: item.histogram_vis_path ? `${API_BASE_URL}${item.histogram_vis_path}` : null,
            }));

            const gtResultsWithPreviews = data.gt_results ? data.gt_results.map(item => ({
                ...item,
                previewUrl: (item.url || item.file_path) ? `${API_BASE_URL}${item.url || item.file_path}` : null,
            })) : [];

            set({ 
                similarityResults: resultsWithPreviews, 
                gtResults: gtResultsWithPreviews,
                queryImageFeatures: data.query_image,
                similarityLoading: false 
            });
            return resultsWithPreviews;

        } catch (error) {
            console.error('Similarity search failed:', error);
            set({ similarityLoading: false, similarityError: error.message });
            get().addToast('error', 'Search failed: ' + error.message);
        }
    },


    /**
     * Select a result for metadata viewing
     */
    selectResult: (result) => {
        set({ selectedResult: result });
    },

    // Import state
    importHistory: [],
    importLoading: false,
    importProgress: 0,
    recomputing: false, // Global recompute state

    // Toast notifications
    toasts: [],


    loadImportHistory: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/images`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const history = await response.json();
            console.log(history);
            // Add preview URLs
            const historyWithPreviews = history.map(item => ({
                ...item,
                previewUrl: (item.url || item.file_path) ? `${API_BASE_URL}${item.url || item.file_path}` : null,
                hogPreviewUrl: item.hog_vis_path ? `${API_BASE_URL}${item.hog_vis_path}` : null,
                huPreviewUrl: item.hu_vis_path ? `${API_BASE_URL}${item.hu_vis_path}` : null,
                cellColorPreviewUrl: item.cell_color_vis_path ? `${API_BASE_URL}${item.cell_color_vis_path}` : null,
                lbpPreviewUrl: item.lbp_vis_path ? `${API_BASE_URL}${item.lbp_vis_path}` : null,
                status: 'completed'
            }));
            console.log(historyWithPreviews);
            set({ importHistory: historyWithPreviews });
        } catch (error) {
            console.error('Failed to load import history:', error);
        }
    },

    /**
     * Select an image
     */
    selectImage: async (imageId) => {
        const image = get().images.find(img => img.id === imageId);
        if (image) {
            set({ selectedImage: image, selectedResult: null });
        }
    },

    /**
     * Set search query
     */
    setSearchQuery: (query) => {
        set({ searchQuery: query });
    },

    /**
     * Set sort option
     */
    setSortBy: (sortBy) => {
        set({ sortBy });
    },

    /**
     * Get filtered and sorted images
     */
    getFilteredImages: () => {
        const { images, searchQuery, sortBy } = get();

        // Filter by search
        let filtered = images;
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = images.filter(img =>
                img.file_name?.toLowerCase().includes(query) ||
                img.id?.toString().includes(query)
            );
        }

        // Sort
        const sorted = [...filtered].sort((a, b) => {
            switch (sortBy) {
                case 'oldest':
                    return new Date(a.created_at) - new Date(b.created_at);
                case 'newest':
                    return new Date(b.created_at) - new Date(a.created_at);
                case 'feature-ready':
                    return (b.brightness ? 1 : 0) - (a.brightness ? 1 : 0);
                default:
                    return 0;
            }
        });

        return sorted;
    },

    /**
     * Set similarity parameters
     */
    setTopK: (topK) => {
        set({ topK });
    },

    setMetric: (metric) => {
        set({ metric });
    },

    /**
     * Find similar images (placeholder - needs backend endpoint)
     */


    /**
     * Select a result for metadata viewing
     */
    selectResult: (result) => {
        set({ selectedResult: result });
    },

    /**
     * Add new image - Upload to real backend
     * Handles File objects from Drag & Drop
     */
    addImage: async (files) => {
        set({ importLoading: true, importProgress: 0 });

        try {
            const fileList = Array.isArray(files) ? files : [files];

            // Create local previews first (instant display)
            const localPreviews = fileList.map(file => ({
                id: 'temp_' + Date.now() + '_' + Math.random(),
                file_name: file.name,
                previewUrl: URL.createObjectURL(file),
                status: 'uploading',
                width: null,
                height: null,
                brightness: null,
                contrast: null
            }));

            set(state => ({
                importHistory: [...localPreviews, ...state.importHistory],
                importProgress: 10
            }));

            // Upload to backend
            const formData = new FormData();
            fileList.forEach(file => {
                formData.append('files', file);
            });

            set({ importProgress: 30 });

            const response = await fetch(`${API_BASE_URL}/api/images/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const serverResults = await response.json();

            set({ importProgress: 90 });

            // Replace temp previews with server data
            set(state => {
                const withoutTemp = state.importHistory.filter(item => !item.id.toString().startsWith('temp_'));
                const newResults = serverResults.map(result => ({
                    ...result,
                    previewUrl: `${API_BASE_URL}${result.url || result.file_path}`,
                    hogPreviewUrl: result.hog_vis_path ? `${API_BASE_URL}${result.hog_vis_path}` : null,
                    huPreviewUrl: result.hu_vis_path ? `${API_BASE_URL}${result.hu_vis_path}` : null,
                    cellColorPreviewUrl: result.cell_color_vis_path ? `${API_BASE_URL}${result.cell_color_vis_path}` : null,
                    lbpPreviewUrl: result.lbp_vis_path ? `${API_BASE_URL}${result.lbp_vis_path}` : null,
                    gaborPreviewUrl: result.gabor_vis_path ? `${API_BASE_URL}${result.gabor_vis_path}` : null,
                    ccvPreviewUrl: result.ccv_vis_path ? `${API_BASE_URL}${result.ccv_vis_path}` : null,
                    status: 'completed'
                }));

                return {
                    importHistory: [...newResults, ...withoutTemp],
                    importLoading: false,
                    importProgress: 100
                };
            });

            // Reset progress after a short delay
            setTimeout(() => set({ importProgress: 0 }), 500);

            get().addToast('success', `Successfully imported ${serverResults.length} images`);

            return serverResults;
        } catch (error) {
            console.error('Import failed:', error);

            // Remove temp items on error
            set(state => ({
                importHistory: state.importHistory.filter(item => !item.id.toString().startsWith('temp_')),
                importLoading: false,
                importProgress: 0
            }));

            get().addToast('error', error.message || 'Failed to add images');
            throw error;
        }
    },

    /**
     * Recompute features for an image
     */
    /**
     * Recompute features for ALL images
     */
    recomputeAllFeatures: async () => {
        set({ recomputing: true });

        try {
            const response = await fetch(`${API_BASE_URL}/api/images/recompute`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`Recompute failed: ${response.statusText}`);
            }

            const updatedAndNewImages = await response.json();

            // Perform a robust update: 
            // 1. Map existing history to update data where IDs match
            // 2. But simpler: Just replace the history with the new data, 
            //    ensuring we keep preview URLs which might be computed client-side in some flows?
            //    Actually the API returns URLs now, so we can just mapped them standardly.

            const historyWithPreviews = updatedAndNewImages.map(item => ({
                ...item,
                previewUrl: (item.url || item.file_path) ? `${API_BASE_URL}${item.url || item.file_path}` : null,
                hogPreviewUrl: item.hog_vis_path ? `${API_BASE_URL}${item.hog_vis_path}` : null,
                huPreviewUrl: item.hu_vis_path ? `${API_BASE_URL}${item.hu_vis_path}` : null,
                cellColorPreviewUrl: item.cell_color_vis_path ? `${API_BASE_URL}${item.cell_color_vis_path}` : null,
                lbpPreviewUrl: item.lbp_vis_path ? `${API_BASE_URL}${item.lbp_vis_path}` : null,
                status: 'completed'
            }));

            set({ importHistory: historyWithPreviews, recomputing: false });
            get().addToast('success', `Successfully recomputed features for ${updatedAndNewImages.length} images`);

        } catch (error) {
            console.error('Recompute all failed:', error);
            set({ recomputing: false });
            get().addToast('error', error.message || 'Failed to recompute features');
        }
    },

    /**
     * Toast notifications
     */
    addToast: (type, message) => {
        const id = Date.now();
        const toast = { id, type, message };
        set((state) => ({ toasts: [...state.toasts, toast] }));

        // Auto-remove after 3 seconds
        setTimeout(() => {
            get().removeToast(id);
        }, 3000);
    },

    removeToast: (id) => {
        set((state) => ({
            toasts: state.toasts.filter(toast => toast.id !== id),
        }));
    },

    /**
     * Evaluation and Optimization
     */
    fetchEvaluation: async (gt = 'clip') => {
        set({ evaluationLoading: true, evaluationError: null });
        try {
            const response = await fetch(`${API_BASE_URL}/api/images/evaluation?gt=${gt}`);
            if (!response.ok) throw new Error(`Failed to fetch evaluation results for ${gt}`);
            const data = await response.json();
            set({ evaluationData: data, evaluationLoading: false });
        } catch (error) {
            set({ evaluationError: error.message, evaluationLoading: false });
        }
    },

    fetchAllEvaluations: async () => {
        set({ evaluationLoading: true, evaluationError: null });
        try {
            const response = await fetch(`${API_BASE_URL}/api/images/evaluation-all`);
            if (!response.ok) throw new Error('Failed to fetch all evaluation results');
            const data = await response.json();
            set({ evaluationAllData: data, evaluationLoading: false });
        } catch (error) {
            set({ evaluationError: error.message, evaluationLoading: false });
        }
    },

    triggerOptimization: async (gt = 'all', trials = 50, allowNegative = false, excludeEmbeddings = false) => {
        set({ isOptimizing: true });
        try {
            const response = await fetch(`${API_BASE_URL}/api/images/optimize?gt=${gt}&trials=${trials}&allow_negative=${allowNegative}&exclude_embeddings=${excludeEmbeddings}`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to start optimization');
            get().addToast('success', `Optimization for ${gt} started in the background.`);
        } catch (error) {
            get().addToast('error', error.message);
            set({ isOptimizing: false });
        }
    },

    fetchWeights: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/images/weights`);
            if (!response.ok) throw new Error('Failed to fetch weights');
            const data = await response.json();
            set({ currentWeights: data });
            return data;
        } catch (error) {
            console.error('Failed to fetch weights:', error);
            // Default empty weights
            set({ currentWeights: {} });
        }
    },
}));

export default useImageStore;
