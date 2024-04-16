class vector {
    constructor(size) {
        this.size = size;
        this.data = Array(this.size).fill(0);
    }

    randomize() {
        for (let i = 0; i < this.size; i++) {
            this.data[i] = Math.random() * 2 - 1;
        }
    }

    map(func) {
        for (let i = 0; i < this.size; i++) {
            this.data[i] = func(this.data[i]);
        }
    }

    sum() {
        return this.data.reduce((sum, value) => sum + value, 0);
    }

    mean() {
        return this.sum() / this.size;
    }

    var() {
        const mean = this.mean();
        return this.data.reduce((sum, value) => sum + (value - mean) ** 2, 0) / this.size;
    }

    sqrt() {
        this.map(Math.sqrt);
    }

    div(n) {
        if (n instanceof vector) {
            if (this.size !== n.size) {
                throw new Error('Sizes of A must match size of B.');
            }

            for (let i = 0; i < this.size; i++) {
                this.data[i] /= n.data[i];
            }
        } else {
            for (let i = 0; i < this.size; i++) {
                this.data[i] /= n;
            }
        }
    }

    add(n) {
        if (n instanceof vector) {
            if (this.size !== n.size) {
                throw new Error('Sizes of A must match size of B.');
            }

            for (let i = 0; i < this.size; i++) {
                this.data[i] += n.data[i];
            }
        } else {
            for (let i = 0; i < this.size; i++) {
                this.data[i] += n;
            }
        }
    }

    subtract(n) {
        if (n instanceof vector) {
            if (this.size !== n.size) {
                throw new Error('Sizes of A must match size of B.');
            }

            for (let i = 0; i < this.size; i++) {
                this.data[i] -= n.data[i];
            }
        } else {
            for (let i = 0; i < this.size; i++) {
                this.data[i] -= n;
            }
        }
    }

    multiply(n) {
        if (n instanceof vector) {
            if (this.size !== n.size) {
                throw new Error('Sizes of A must match size of B.');
            }

            for (let i = 0; i < this.size; i++) {
                this.data[i] *= n.data[i];
            }
        } else {
            for (let i = 0; i < this.size; i++) {
                this.data[i] *= n;
            }

        }
    }

    update(learningRate) {
        this.map(value => value * learningRate);
    }

    get(index) {
        return this.data[index];
    }

    set(index, value) {
        this.data[index] = value;
    }

    assign(array) {
        for (let i = 0; i < this.size; i++) {
            this.data[i] = array[i];
        }
    }
}
