import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';

class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      formData: {
        A: 0,
        C: 0,
        D: 0,
        G: 0,
        H: 0,
        I: 0,
        J: 0,
        K: 0,
        N: 0,
        O: 0,
        P: 0,
        S: 0,
        U: 0,
        V: 0,
        Y: 0,
        Z: 0
      },
      result: ""
    };
  }

  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData
    });
  }

  handlePredictClick = (event) => {
    const formData = this.state.formData;
    this.setState({ isLoading: true });
    fetch('http://127.0.0.1:5000/prediction/',
      {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify(formData)
      })
      .then(response => response.json())
      .then(response => {
        this.setState({
          result: response.result,
          isLoading: false
        });
      });
  }

  handleCancelClick = (event) => {
    this.setState({ result: "" });
  }

  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    const result = this.state.result;

    var A = []
    for (var i = 0; i <= 10; i = +(i + 0.1).toFixed(1)) {
      A.push(<option key = {i} value = {i}>{i}</option>);
    }
    var C = []
    for (var i = 0; i <= 4; i = +(i + 0.1).toFixed(1)) {
      C.push(<option key = {i} value = {i}>{i}</option>);
    }
    var D = []
    for (var i = 0; i <= 4; i = +(i + 0.1).toFixed(1)){
      D.push(<option key = {i} value = {i}>{i}</option>);
    }
    var G = []
    for (var i = 0.1; i <= 3; i = +(i + 0.1).toFixed(1)) {
      G.push(<option key = {i} value = {i}>{i}</option>);
    }
    var H = []
    for (var i = 1; i <= 30; i = +(i + 0.1).toFixed(1)) {
      H.push(<option key = {i} value = {i}>{i}</option>);
    }
    var I = []
    for (var i = 0; i <= 5; i = +(i + 0.1).toFixed(1)) {
      I.push(<option key = {i} value = {i}>{i}</option>);
    }
    var J = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      H.push(<option key = {i} value = {i}>{i}</option>);
    }
    var K = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      K.push(<option key = {i} value = {i}>{i}</option>);
    }
    var N = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      N.push(<option key = {i} value = {i}>{i}</option>);
    }
    var O = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      O.push(<option key = {i} value = {i}>{i}</option>);
    }
    var P = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      P.push(<option key = {i} value = {i}>{i}</option>);
    }
    var S = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      S.push(<option key = {i} value = {i}>{i}</option>);
    }
    var U = []
    for (var i = 0; i <= 3; i = +(i + 0.1).toFixed(1)) {
      U.push(<option key = {i} value = {i}>{i}</option>);
    }
    var V = []
    for (var i = 0; i <= 6; i = +(i + 0.1).toFixed(1)) {
      V.push(<option key = {i} value = {i}>{i}</option>);
    }
    var Y = []
    for (var i = 0; i <= 10; i = +(i + 0.1).toFixed(1)) {
      Y.push(<option key = {i} value = {i}>{i}</option>);
    }
    var Z = []
    for (var i = 0; i <= 30; i = +(i + 0.1).toFixed(1)) {
      Z.push(<option key = {i} value = {i}>{i}</option>);
    }

    return (
      <Container>
        <div>
          <h1 className="title">Language Level Classifier</h1>
        </div>
        <div className="content">
          <Form>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>A</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.A}
                  name="A"
                  onChange={this.handleChange}>
                  {A}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>C</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.C}
                  name="C"
                  onChange={this.handleChange}>
                  {C}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>D</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.D}
                  name="D"
                  onChange={this.handleChange}>
                  {D}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>G</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.G}
                  name="G"
                  onChange={this.handleChange}>
                  {G}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>H</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.H}
                  name="H"
                  onChange={this.handleChange}>
                  {H}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>I</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.I}
                  name="I"
                  onChange={this.handleChange}>
                  {I}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>K</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.K}
                  name="K"
                  onChange={this.handleChange}>
                  {K}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>N</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.N}
                  name="N"
                  onChange={this.handleChange}>
                  {N}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>O</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.O}
                  name="O"
                  onChange={this.handleChange}>
                  {O}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>P</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.P}
                  name="P"
                  onChange={this.handleChange}>
                  {P}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>S</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.S}
                  name="S"
                  onChange={this.handleChange}>
                  {S}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>U</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.U}
                  name="U"
                  onChange={this.handleChange}>
                  {U}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Form.Row>
              <Form.Group as={Col}>
                <Form.Label>Y</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.Y}
                  name="Y"
                  onChange={this.handleChange}>
                  {Y}
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}>
                <Form.Label>Z</Form.Label>
                <Form.Control
                  as="select"
                  value={formData.Z}
                  name="Z"
                  onChange={this.handleChange}>
                  {Z}
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Row>
              <Col>
                <Button
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}>
                  { isLoading ? 'Making prediction' : 'Predict' }
                </Button>
              </Col>
              <Col>
                <Button
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}>
                  Reset prediction
                </Button>
              </Col>
            </Row>
          </Form>
          {result === "" ? null :
            (<Row>
              <Col className="result-container">
                <h5 id="result">{result}</h5>
              </Col>
            </Row>)
          }
        </div>
      </Container>
    );
  }
}

export default App;
