let x_vals = [];
let y_vals = [];

let a, b, c;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(window.innerHeight, window.innerHeight-100);
  function randomStart(){return random(2)-1; }
  a = tf.variable(tf.scalar(randomStart()));
  b = tf.variable(tf.scalar(randomStart()));
  c = tf.variable(tf.scalar(randomStart()));
}

function mapX(x){
    return map(x, -1, 1, 0, width);
}
function mapY(y){
    return map(y, -1, 1, height, 0);
}

function loss(pred, given) {
  return pred.sub(given).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^2 + bx + c;
  const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
  return ys;
}

function mousePressed() {
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {

  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = mapX(x_vals[i]);
    let py = mapY(y_vals[i]);
    point(px, py);
  }


  const curveX = [];
  for(let x = -1; x < 1.01; x += 0.05){
    curveX.push(x);
  }

  const tcurveY = tf.tidy(() => predict(curveX));
  let curveY = tcurveY.dataSync();
  tcurveY.dispose();

  for(var i = 1; i < curveX.length; i++){
      strokeWeight(2);
      line(mapX(curveX[i-1]),mapY(curveY[i-1]),mapX(curveX[i]),mapY(curveY[i]));
  }
  console.log(tf.memory().numTensors);
  //noLoop();
}
