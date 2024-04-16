class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = [];

        for (let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = 0;
            }
        }
    }

    subtract(b) {
        const result = new Matrix(this.rows, this.cols);

        if (this.rows !== b.rows || this.cols !== b.cols) {
            throw new Error("Invalid matrix dimensions for subtraction");
        }

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] - b.data[i][j];
            }
        }

        return result;
    }

    set(row, col, value) {
        this.data[row][col] = value;
    }

    get(row, col) {
        return this.data[row][col];
    }

    static multiply(a, b) {
        if (a.cols !== b.cols || a.rows !== b.rows) {
            throw new Error("Invalid matrix dimensions for multiplication");
        }

        const result = new Matrix(a.rows, b.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }

        return result;
    }

    normalize() {
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] /= sum;
            }
        }
    }

    activate(activationFunction) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = activationFunction(this.data[i][j]);
            }
        }
    }

    transpose() {
        const result = new Matrix(this.cols, this.rows);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }

        return result;
    }

    static concatenate(matrices) {
        const rows = matrices[0].rows;
        const cols = matrices.reduce((acc, matrix) => acc + matrix.cols, 0);

        const result = new Matrix(rows, cols);

        let currentCol = 0;
        for (const matrix of matrices) {
            for (let i = 0; i < matrix.rows; i++) {
                for (let j = 0; j < matrix.cols; j++) {
                    result.data[i][currentCol + j] = matrix.data[i][j];
                }
            }
            currentCol += matrix.cols;
        }

        return result;
    }

    static multiplyElementWise(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Invalid matrix dimensions for element-wise multiplication");
        }

        const result = new Matrix(a.rows, a.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }

        return result;
    }

    static dot(a, b) {
        if (a.cols !== b.rows) {
            throw new Error("Invalid matrix dimensions for dot product");
        }

        const result = new Matrix(a.rows, b.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    dot(b) {
        if (this.cols !== b.rows) {
            throw new Error("Invalid matrix dimensions for dot product");
        }

        const result = new Matrix(this.rows, b.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    add (b) {
        if (this.rows !== b.rows || this.cols !== b.cols) {
            throw new Error("Invalid matrix dimensions for addition");
        }

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] += b.data[i][j];
            }
        }
    }

    split(axis, start, end) {
        if (axis === 0) {
            const numRows = end - start + 1;
            const result = new Matrix(numRows, this.cols);

            for (let i = 0; i < numRows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    result.data[i][j] = this.data[start + i][j];
                }
            }

            return result;
        } else if (axis === 1) {
            const numCols = end - start + 1;
            const result = new Matrix(this.rows, numCols);

            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < numCols; j++) {
                    result.data[i][j] = this.data[i][start + j];
                }
            }

            return result;
        } else {
            throw new Error("Invalid axis specified");
        }
    }

    static add(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Invalid matrix dimensions for addition");
        }

        const result = new Matrix(a.rows, a.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }

        return result;
    }

    

    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Invalid matrix dimensions for subtraction");
        }

        const result = new Matrix(a.rows, a.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }

        return result;
    }

    applyActivation(activationFunction) {
        const result = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = activationFunction(this.data[i][j]);
            }
        }

        return result;
    }

    applyActivationDerivative(activationFunctionDerivative) {
        const result = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = activationFunctionDerivative(this.data[i][j]);
            }
        }

        return result;
    }

    multiplyScaler(value) {
        const result = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
              
                result.data[i][j] = this.data[i][j] * value;
            }
        }

        return result;
    }

    
    static randomize(rows, cols) {
        const result = new Matrix(rows, cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = Math.random();
            }
        }

        return result;
    }

    static zeros(rows, cols) {
        const result = new Matrix(rows, cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = 0;
            }
        }

        return result;
    }
}
