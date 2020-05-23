import Vue from 'vue'
import Router from 'vue-router'
import graph from '@/components/graph'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'graph',
      component: graph
    }
  ]
})
