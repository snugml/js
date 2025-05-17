// cluster.mjs

export class KMeans {
    constructor(n_clusters = 3, max_iter = 300, tolerance = 1e-4, random_state = null) {
        this.n_clusters = n_clusters;
        this.max_iter = max_iter;
        this.tolerance = tolerance;
        this.centroids = [];
        this.labels = [];
        this.isFit = false;

        if (random_state !== null) {
            this._random = this.seededRandom(random_state);
        } else {
            this._random = Math.random;
        }
    }

    euclideanDistance(point1, point2) {
        return Math.sqrt(point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0));
    }

    initializeCentroids(X) {
        const centroids = [];
        const usedIndices = new Set();
        while (centroids.length < this.n_clusters) {
            const idx = Math.floor(this._random() * X.length);
            if (!usedIndices.has(idx)) {
                centroids.push([...X[idx]]);
                usedIndices.add(idx);
            }
        }
        return centroids;
    }

    assignClusters(X, centroids) {
        return X.map(point => {
            let minDist = Infinity;
            let cluster = 0;
            for (let i = 0; i < centroids.length; i++) {
                const dist = this.euclideanDistance(point, centroids[i]);
                if (dist < minDist) {
                    minDist = dist;
                    cluster = i;
                }
            }
            return cluster;
        });
    }

    updateCentroids(X, labels) {
        const newCentroids = Array(this.n_clusters).fill(0).map(() => Array(X[0].length).fill(0));
        const counts = Array(this.n_clusters).fill(0);

        for (let i = 0; i < X.length; i++) {
            const cluster = labels[i];
            counts[cluster]++;
            for (let j = 0; j < X[i].length; j++) {
                newCentroids[cluster][j] += X[i][j];
            }
        }

        for (let i = 0; i < this.n_clusters; i++) {
            if (counts[i] === 0) continue; // evitar dividir por cero
            for (let j = 0; j < newCentroids[i].length; j++) {
                newCentroids[i][j] /= counts[i];
            }
        }

        return newCentroids;
    }

    centroidsShift(oldCentroids, newCentroids) {
        let totalShift = 0;
        for (let i = 0; i < oldCentroids.length; i++) {
            totalShift += this.euclideanDistance(oldCentroids[i], newCentroids[i]);
        }
        return totalShift;
    }

    fit(X, y = null) {
        try {
            if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) {
                throw new Error('X must be a non-empty 2D array.');
            }

            this.centroids = this.initializeCentroids(X);

            for (let iter = 0; iter < this.max_iter; iter++) {
                const labels = this.assignClusters(X, this.centroids);
                const newCentroids = this.updateCentroids(X, labels);
                const shift = this.centroidsShift(this.centroids, newCentroids);

                this.centroids = newCentroids;
                this.labels = labels;

                if (shift < this.tolerance) break;
            }

            this.isFit = true;
        } catch (error) {
            console.error(error.message);
        }
    }

    predict(X) {
        if (!this.isFit) {
            console.warn("Model has not been fitted yet.");
            return [];
        }

        return this.assignClusters(X, this.centroids);
    }

    // Método opcional para obtener los centroides
    getCentroids() {
        return this.centroids;
    }

    // Método opcional para obtener etiquetas después del fit
    getLabels() {
        return this.labels;
    }

    seededRandom(seed) {
        let m = 2 ** 31 - 1;
        let a = 1103515245;
        let c = 12345;
        let state = seed % m;
        return function () {
            state = (a * state + c) % m;
            return state / m;
        };
    }
}
