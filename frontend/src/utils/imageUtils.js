/**
 * Get image URL (handle both absolute and relative paths)
 */
export function getImageUrl(fileUrl) {
    if (!fileUrl) return '';

    // If already absolute URL, return as-is
    if (fileUrl.startsWith('http://') || fileUrl.startsWith('https://')) {
        return fileUrl;
    }

    // If relative, ensure it starts with /
    return fileUrl.startsWith('/') ? fileUrl : `/${fileUrl}`;
}

/**
 * Generate placeholder image URL
 */
export function getPlaceholderImage(width = 400, height = 300, text = '') {
    const encodedText = encodeURIComponent(text);
    return `https://via.placeholder.com/${width}x${height}/1e293b/64748b?text=${encodedText}`;
}

/**
 * Validate image file
 */
export function isValidImageFile(filename) {
    const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'];
    const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
    return validExtensions.includes(ext);
}

/**
 * Extract filename from path
 */
export function getFilename(path) {
    return path.split('/').pop() || path;
}

export default {
    getImageUrl,
    getPlaceholderImage,
    isValidImageFile,
    getFilename,
};
