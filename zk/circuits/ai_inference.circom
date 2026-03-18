pragma circom 2.0.0;

template AIInference() {

    signal input x1;
    signal input x2;

    signal input w1;
    signal input w2;

    signal output y;

    y <== x1 * w1 + x2 * w2;
}

component main = AIInference();