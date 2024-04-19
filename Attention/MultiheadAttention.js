
class MultiheadAttention {
    constructor(dModel, numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headSize = dModel / numHeads;

        const value = 1/ (dModel * this.numHeads);
        this.queryWeights = Matrix.matrixValue(value, dModel, dModel)
        this.keyWeights = Matrix.matrixValue(value, dModel, dModel);
        this.valueWeights = Matrix.matrixValue(value, dModel, dModel);
        this.outputWeights = Matrix.matrixValue(value, dModel, dModel);

    }

    forward(query, key, value) {
        const batchSize = query.rows;

        this.outputs = [];
        this.inputs = [[query, key, value]];

        const queryProjected = query.dot(this.queryWeights);
        const keyProjected = key.dot(this.keyWeights);
        const valueProjected = value.dot(this.valueWeights);

        var scaledDotProducts = new Matrix(0, this.dModel);
        for (let i = 0; i < this.numHeads; i++) {
            const scaledDotProduct = this.scaledDotProductAttention(
                queryProjected.getRow(i),
                keyProjected.getRow(i),
                valueProjected.getRow(i),
            );

            scaledDotProducts = scaledDotProducts.append(scaledDotProduct);
            this.outputs.push(scaledDotProduct);
        }

        // const concatenated = Matrix.concatenate(scaledDotProducts);
        const output = scaledDotProducts.dot(this.outputWeights);
        this.outputs.push(output);
        return output;
    }

    backward(input, target, output, loss, learningRate) {
        const gradientsBackwardOutput = Matrix.subtract(target, output);
        gradientsBackwardOutput.normalize();

        const gradientsBackwardInput = gradientsBackwardOutput.dot(this.outputWeights.transpose());
        const inputTranspose = output.transpose();
        const gradientsOutputWeights = inputTranspose.dot(gradientsBackwardOutput);
        this.outputWeights.add(gradientsOutputWeights.multiplyScaler(learningRate));

        const gradientsBackwardInputWeights = [];
        for (let i = 0; i < this.numHeads; i++) {
            var query = (this.queryWeights.getRow(i)).transpose();
            query = query.dot(gradientsBackwardInput.getRow(i));
            var key = (this.keyWeights.getRow(i)).transpose();
            key = key.dot(gradientsBackwardInput.getRow(i));
            var value = (this.valueWeights.getRow(i)).transpose();
            value = value.dot(gradientsBackwardInput.getRow(i));

            query.add(key);
            query.add(value);
            
            gradientsBackwardInputWeights.push(query);
        }

        for (let i = 0; i < this.numHeads; i++) {
            this.queryWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
            this.keyWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
            this.valueWeights.add(gradientsBackwardInputWeights[i].multiplyScaler(learningRate));
        }
    }

    scaledDotProductAttention(query, key, value) {
        const sqrtHeadSize = Math.sqrt(this.headSize);

        const scores = query.dot(key.transpose()).multiplyScaler(1 / sqrtHeadSize);
        const attentionWeights = scores.softmax();

        const context = attentionWeights.dot(value);

        return context;
    }
}