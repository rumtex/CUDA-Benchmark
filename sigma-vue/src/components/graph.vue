<template lang="pug">
.ct(@click="sig")
  input#filein(type="file" @change="handleFile")
  #graph
</template>

<script>
import {sigma} from 'sigma-webpack'

export default {
  name: 'graph',
  data () {
    return {
      msg: 'Welcome to Your Vue.js App',
      graph: null,
      reader: null,
    }
  },
  mounted() {
    this.reader = new FileReader();
    this.reader.onload = ({target}) => {
      this.createGraph(JSON.parse(target.result))
    };
    this.createGraph({nodes:[{id:'0',label:'1.0',x:1,y:1,size:1}]})

    window.sigma = sigma
    var newScript = document.createElement('script');
    newScript.src = '/static/sigma.renderers.edgeLabels.min.js';
    window.document.head.appendChild(newScript);
  },
  methods: {
    sig() {
      console.log(sigma.plugins)
    },
    handleFile({target}) {
      this.reader.readAsText(target.files[0]);
    },
    createGraph(jsonka) {
      this.graph && this.graph.kill()
      this.graph = new sigma({
        graph: jsonka,
        settings: {
          defaultNodeColor: '#ec5148',
          type: 'canvas',
          edgeLabelSize: 'proportional'
        },
        container: 'graph'
      });
    }
  }
}
</script>

<style lang="stylus" scoped>
.ct
  #graph
    text-align initial
    width 100%
    height calc(100vh - 100px)
    position relative
    background #f4fbed

h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
