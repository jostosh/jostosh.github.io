(window.webpackJsonp=window.webpackJsonp||[]).push([[6],{153:function(t,e,a){"use strict";a.r(e),a.d(e,"pageQuery",function(){return m});a(34);var n=a(7),A=a.n(n),r=a(0),i=a.n(r),o=a(157),c=a(167),l=a(162),s=a(163),d=a(158),u=a(195);a(151);var p=function(t){function e(){return t.apply(this,arguments)||this}return A()(e,t),e.prototype.render=function(){var t=this.props.data.markdownRemark,e=this.props.data.site.siteMetadata.title,a=this.props.pageContext,n=a.previous,A=a.next,r={identifier:t.id,title:t.frontmatter.title};return i.a.createElement(l.a,{location:this.props.location,title:e},i.a.createElement(s.a,{title:t.frontmatter.title,description:t.frontmatter.description||t.excerpt}),i.a.createElement("h1",null,t.frontmatter.title),i.a.createElement("p",{style:Object.assign({},Object(d.b)(-.1),{display:"block",marginBottom:Object(d.a)(1),marginTop:Object(d.a)(-.5),fontStyle:"italic"})},t.frontmatter.date),i.a.createElement("div",{dangerouslySetInnerHTML:{__html:t.html}}),i.a.createElement("hr",{style:{marginBottom:Object(d.a)(1)}}),i.a.createElement(c.a,null),i.a.createElement("ul",{style:{display:"flex",flexWrap:"wrap",justifyContent:"space-between",listStyle:"none",padding:0}},i.a.createElement("li",null,n&&i.a.createElement(o.a,{to:n.fields.slug,rel:"prev"},"← ",n.frontmatter.title)),i.a.createElement("li",null,A&&i.a.createElement(o.a,{to:A.fields.slug,rel:"next"},A.frontmatter.title," →"))),i.a.createElement(u.DiscussionEmbed,{shortname:"https-jostosh-github-io",config:r}))},e}(i.a.Component);e.default=p;var m="3654438753"},157:function(t,e,a){"use strict";a.d(e,"b",function(){return s});var n=a(0),A=a.n(n),r=a(4),i=a.n(r),o=a(33),c=a.n(o);a.d(e,"a",function(){return c.a});a(159);var l=A.a.createContext({}),s=function(t){return A.a.createElement(l.Consumer,null,function(e){return t.data||e[t.query]&&e[t.query].data?(t.render||t.children)(t.data?t.data.data:e[t.query].data):A.a.createElement("div",null,"Loading (StaticQuery)")})};s.propTypes={data:i.a.object,query:i.a.string.isRequired,render:i.a.func,children:i.a.func}},158:function(t,e,a){"use strict";a.d(e,"a",function(){return c}),a.d(e,"b",function(){return l});var n=a(172),A=a.n(n),r=a(173),i=a.n(r);i.a.overrideThemeStyles=function(){return{"a.gatsby-resp-image-link":{boxShadow:"none",textDecoration:"none",border:"none",showCaptions:!0},"h1,h2":{color:"#ff8c00"},"h3,h4,h5,h6":{color:"#cc7000"},a:{textDecoration:"none",textShadow:"none",borderBottom:"1px solid #ff8c00",backgroundImage:"none",color:"#C35C00"},em:{color:"#cc7000",fontStyle:"normal"}}};var o=new A.a(i.a);var c=o.rhythm,l=o.scale},159:function(t,e,a){var n;t.exports=(n=a(161))&&n.default||n},161:function(t,e,a){"use strict";a.r(e);a(34);var n=a(0),A=a.n(n),r=a(4),i=a.n(r),o=a(56),c=a(2),l=function(t){var e=t.location,a=c.default.getResourcesForPathnameSync(e.pathname);return a?A.a.createElement(o.a,Object.assign({location:e,pageResources:a},a.json)):null};l.propTypes={location:i.a.shape({pathname:i.a.string.isRequired}).isRequired},e.default=l},162:function(t,e,a){"use strict";a(34);var n=a(7),A=a.n(n),r=a(0),i=a.n(r),o=a(157),c=a(158),l=function(t){function e(){return t.apply(this,arguments)||this}return A()(e,t),e.prototype.render=function(){var t,e=this.props,a=e.location,n=e.title,A=e.children;return t="/"===a.pathname?i.a.createElement("h1",{style:Object.assign({},Object(c.b)(1.5),{marginBottom:Object(c.a)(1.5),marginTop:0,color:"#ff8c00"})},i.a.createElement(o.a,{style:{boxShadow:"none",textDecoration:"none"},to:"/"},n)):i.a.createElement("h3",{style:{fontFamily:"Montserrat, sans-serif",marginTop:0,color:"#ff8c00"}},i.a.createElement(o.a,{style:{boxShadow:"none",textDecoration:"none"},to:"/"},n)),i.a.createElement("div",{style:{marginLeft:"auto",marginRight:"auto",maxWidth:Object(c.a)(30),padding:Object(c.a)(1.5)+" "+Object(c.a)(.75)}},i.a.createElement("header",null,t),i.a.createElement("main",null,A))},e}(i.a.Component);e.a=l},163:function(t,e,a){"use strict";var n=a(164),A=a(0),r=a.n(A),i=a(4),o=a.n(i),c=a(174),l=a.n(c);function s(t){var e=t.description,a=t.lang,A=t.meta,i=t.title,o=n.data.site,c=e||o.siteMetadata.description;return r.a.createElement(l.a,{htmlAttributes:{lang:a},title:i,titleTemplate:"%s | "+o.siteMetadata.title,meta:[{name:"description",content:c},{property:"og:title",content:i},{property:"og:description",content:c},{property:"og:type",content:"website"},{name:"twitter:card",content:"summary"},{name:"twitter:creator",content:o.siteMetadata.author},{name:"twitter:title",content:i},{name:"twitter:description",content:c}].concat(A)})}s.defaultProps={lang:"en",meta:[],description:""},s.propTypes={description:o.a.string,lang:o.a.string,meta:o.a.arrayOf(o.a.object),title:o.a.string.isRequired},e.a=s},164:function(t){t.exports={data:{site:{siteMetadata:{title:"Machine Learning Blog - JvdW",description:"A blog discussing various topic in Machine Learning.",author:"Jos van de Wolfshaar"}}}}},167:function(t,e,a){"use strict";a(168);var n=a(170),A=a(0),r=a.n(A),i=a(157),o=a(171),c=a.n(o),l=a(158);var s="4007731267";e.a=function(){return r.a.createElement(i.b,{query:s,render:function(t){var e=t.site.siteMetadata,a=e.author;return e.social,r.a.createElement("div",{style:{display:"flex",marginBottom:Object(l.a)(2.5)}},r.a.createElement(c.a,{fixed:t.avatar.childImageSharp.fixed,alt:a,style:{marginRight:Object(l.a)(.5),marginBottom:0,minWidth:50,borderRadius:"100%"},imgStyle:{borderRadius:"50%"}}),r.a.createElement("p",null,"Written by ",r.a.createElement("strong",null,a),", a Machine Learning practitioner living and working in Amsterdam."," "))},data:n})}},170:function(t){t.exports={data:{avatar:{childImageSharp:{fixed:{base64:"data:image/jpeg;base64,/9j/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wgARCAAUABQDASIAAhEBAxEB/8QAGAABAAMBAAAAAAAAAAAAAAAAAAIDBAX/xAAWAQEBAQAAAAAAAAAAAAAAAAACAAH/2gAMAwEAAhADEAAAAb8/TqzdyRHlyGbRX//EABsQAAICAwEAAAAAAAAAAAAAAAABAhIDBBEi/9oACAEBAAEFAtlsooTMnKYuXH6EizR//8QAFhEAAwAAAAAAAAAAAAAAAAAAASAx/9oACAEDAQE/ARE//8QAFREBAQAAAAAAAAAAAAAAAAAAASD/2gAIAQIBAT8BY//EABsQAAIDAQEBAAAAAAAAAAAAAAECABAxIRFB/9oACAEBAAY/AkTA2xShO16fk7Xa2f/EABsQAAIDAQEBAAAAAAAAAAAAAAERACFRMRCB/9oACAEBAAE/IWAEkWGZLbiBRQdDiUPoPfG6kwSuAJAp/9oADAMBAAIAAwAAABAMGIL/xAAWEQEBAQAAAAAAAAAAAAAAAAARABD/2gAIAQMBAT8QCxm//8QAFxEAAwEAAAAAAAAAAAAAAAAAAAEREP/aAAgBAgEBPxBK8hEf/8QAHhAAAgICAgMAAAAAAAAAAAAAAREAIUFRMXEQ4fD/2gAIAQEAAT8QfJTPIe0DpgwsgleDkAYCZ25yPDAr44oy9JdDqEPbuCAYa5n/2Q==",width:50,height:50,src:"/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/c15d6/profile-pic.jpg",srcSet:"/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/c15d6/profile-pic.jpg 1x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/43c20/profile-pic.jpg 1.5x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/da97e/profile-pic.jpg 2x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/3e75a/profile-pic.jpg 3x"}}},site:{siteMetadata:{author:"Jos van de Wolfshaar",social:{twitter:"josvdwolfshaar"}}}}}}}]);
//# sourceMappingURL=component---src-templates-blog-post-js-9180e7f88c8154bda3b8.js.map