(window.webpackJsonp=window.webpackJsonp||[]).push([[2],{157:function(e,t,a){"use strict";a.d(t,"b",function(){return c});var i=a(0),r=a.n(i),n=a(4),o=a.n(n),s=a(33),d=a.n(s);a.d(t,"a",function(){return d.a});a(159);var l=r.a.createContext({}),c=function(e){return r.a.createElement(l.Consumer,null,function(t){return e.data||t[e.query]&&t[e.query].data?(e.render||e.children)(e.data?e.data.data:t[e.query].data):r.a.createElement("div",null,"Loading (StaticQuery)")})};c.propTypes={data:o.a.object,query:o.a.string.isRequired,render:o.a.func,children:o.a.func}},158:function(e,t,a){"use strict";a.d(t,"a",function(){return d}),a.d(t,"b",function(){return l});var i=a(165),r=a.n(i),n=a(166),o=a.n(n);o.a.overrideThemeStyles=function(){return{"a.gatsby-resp-image-link":{boxShadow:"none",textDecoration:"none",border:"none",showCaptions:!0},"h1,h2":{color:"#ff8c00"},"h3,h4,h5,h6":{color:"#cc7000"},a:{textDecoration:"none",textShadow:"none",backgroundImage:"none",color:"#C35C00"},em:{color:"#cc7000",fontStyle:"normal"}}};var s=new r.a(o.a);var d=s.rhythm,l=s.scale},159:function(e,t,a){var i;e.exports=(i=a(162))&&i.default||i},160:function(e,t,a){"use strict";a(34);var i=a(7),r=a.n(i),n=a(0),o=a.n(n),s=a(157),d=a(158),l=function(e){function t(){return e.apply(this,arguments)||this}return r()(t,e),t.prototype.render=function(){var e,t=this.props,a=t.location,i=t.title,r=t.children;return e="/"===a.pathname?o.a.createElement("h1",{style:Object.assign({},Object(d.b)(1.5),{marginBottom:Object(d.a)(1.5),marginTop:0,color:"#ff8c00"})},o.a.createElement(s.a,{style:{boxShadow:"none",textDecoration:"none"},to:"/"},i)):o.a.createElement("h3",{style:{fontFamily:"Montserrat, sans-serif",marginTop:0,color:"#ff8c00"}},o.a.createElement(s.a,{style:{boxShadow:"none",textDecoration:"none"},to:"/"},i)),o.a.createElement("div",{style:{marginLeft:"auto",marginRight:"auto",maxWidth:Object(d.a)(30),padding:Object(d.a)(1.5)+" "+Object(d.a)(.75)}},o.a.createElement("header",null,e),o.a.createElement("main",null,r))},t}(o.a.Component);t.a=l},161:function(e,t,a){"use strict";var i=a(163),r=a(0),n=a.n(r),o=a(4),s=a.n(o),d=a(168),l=a.n(d);function c(e){var t=e.description,a=e.lang,r=e.meta,o=e.title,s=i.data.site,d=t||s.siteMetadata.description;return n.a.createElement(l.a,{htmlAttributes:{lang:a},title:o,titleTemplate:"%s | "+s.siteMetadata.title,meta:[{name:"description",content:d},{property:"og:title",content:o},{property:"og:description",content:d},{property:"og:type",content:"website"},{name:"twitter:card",content:"summary"},{name:"twitter:creator",content:s.siteMetadata.author},{name:"twitter:title",content:o},{name:"twitter:description",content:d}].concat(r)})}c.defaultProps={lang:"en",meta:[],description:""},c.propTypes={description:s.a.string,lang:s.a.string,meta:s.a.arrayOf(s.a.object),title:s.a.string.isRequired},t.a=c},162:function(e,t,a){"use strict";a.r(t);a(34);var i=a(0),r=a.n(i),n=a(4),o=a.n(n),s=a(56),d=a(2),l=function(e){var t=e.location,a=d.default.getResourcesForPathnameSync(t.pathname);return a?r.a.createElement(s.a,Object.assign({location:t,pageResources:a},a.json)):null};l.propTypes={location:o.a.shape({pathname:o.a.string.isRequired}).isRequired},t.default=l},163:function(e){e.exports={data:{site:{siteMetadata:{title:"Machine Learning Blog - JvdW",description:"A blog discussing various topic in Machine Learning.",author:"Jos van de Wolfshaar"}}}}},164:function(e,t,a){"use strict";a(170);var i=a(172),r=a(0),n=a.n(r),o=a(157),s=a(173),d=a.n(s),l=a(158);var c="4007731267";t.a=function(){return n.a.createElement(o.b,{query:c,render:function(e){var t=e.site.siteMetadata,a=t.author;return t.social,n.a.createElement("div",{style:{display:"flex",marginBottom:Object(l.a)(2.5)}},n.a.createElement(d.a,{fixed:e.avatar.childImageSharp.fixed,alt:a,style:{marginRight:Object(l.a)(.5),marginBottom:0,minWidth:50,borderRadius:"100%"},imgStyle:{borderRadius:"50%"}}),n.a.createElement("p",null,"Written by ",n.a.createElement("strong",null,a),", a Machine Learning practitioner living and working in Amsterdam."," "))},data:i})}},170:function(e,t,a){"use strict";a(171)("fixed",function(e){return function(){return e(this,"tt","","")}})},171:function(e,t,a){var i=a(11),r=a(18),n=a(19),o=/"/g,s=function(e,t,a,i){var r=String(n(e)),s="<"+t;return""!==a&&(s+=" "+a+'="'+String(i).replace(o,"&quot;")+'"'),s+">"+r+"</"+t+">"};e.exports=function(e,t){var a={};a[e]=t(s),i(i.P+i.F*r(function(){var t=""[e]('"');return t!==t.toLowerCase()||t.split('"').length>3}),"String",a)}},172:function(e){e.exports={data:{avatar:{childImageSharp:{fixed:{base64:"data:image/jpeg;base64,/9j/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wgARCAAUABQDASIAAhEBAxEB/8QAGAABAAMBAAAAAAAAAAAAAAAAAAIDBAX/xAAWAQEBAQAAAAAAAAAAAAAAAAACAAH/2gAMAwEAAhADEAAAAb8/TqzdyRHlyGbRX//EABsQAAICAwEAAAAAAAAAAAAAAAABAhIDBBEi/9oACAEBAAEFAtlsooTMnKYuXH6EizR//8QAFhEAAwAAAAAAAAAAAAAAAAAAASAx/9oACAEDAQE/ARE//8QAFREBAQAAAAAAAAAAAAAAAAAAASD/2gAIAQIBAT8BY//EABsQAAIDAQEBAAAAAAAAAAAAAAECABAxIRFB/9oACAEBAAY/AkTA2xShO16fk7Xa2f/EABsQAAIDAQEBAAAAAAAAAAAAAAERACFRMRCB/9oACAEBAAE/IWAEkWGZLbiBRQdDiUPoPfG6kwSuAJAp/9oADAMBAAIAAwAAABAMGIL/xAAWEQEBAQAAAAAAAAAAAAAAAAARABD/2gAIAQMBAT8QCxm//8QAFxEAAwEAAAAAAAAAAAAAAAAAAAEREP/aAAgBAgEBPxBK8hEf/8QAHhAAAgICAgMAAAAAAAAAAAAAAREAIUFRMXEQ4fD/2gAIAQEAAT8QfJTPIe0DpgwsgleDkAYCZ25yPDAr44oy9JdDqEPbuCAYa5n/2Q==",width:50,height:50,src:"/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/c15d6/profile-pic.jpg",srcSet:"/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/c15d6/profile-pic.jpg 1x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/43c20/profile-pic.jpg 1.5x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/da97e/profile-pic.jpg 2x,\n/static/5d8f6ea1d4c4a0e40fc83692a4515ca6/3e75a/profile-pic.jpg 3x"}}},site:{siteMetadata:{author:"Jos van de Wolfshaar",social:{twitter:"josvdwolfshaar"}}}}}},173:function(e,t,a){"use strict";var i=a(8);t.__esModule=!0,t.default=void 0;var r,n=i(a(7)),o=i(a(35)),s=i(a(75)),d=i(a(76)),l=i(a(0)),c=i(a(4)),u=function(e){var t=(0,d.default)({},e);return t.resolutions&&(t.fixed=t.resolutions,delete t.resolutions),t.sizes&&(t.fluid=t.sizes,delete t.sizes),t},A=Object.create({}),f=function(e){var t=u(e),a=t.fluid?t.fluid.src:t.fixed.src;return A[a]||!1},p=new WeakMap;var g=function(e,t){var a=(void 0===r&&"undefined"!=typeof window&&window.IntersectionObserver&&(r=new window.IntersectionObserver(function(e){e.forEach(function(e){if(p.has(e.target)){var t=p.get(e.target);(e.isIntersecting||e.intersectionRatio>0)&&(r.unobserve(e.target),p.delete(e.target),t())}})},{rootMargin:"200px"})),r);return a&&(a.observe(e),p.set(e,t)),function(){a.unobserve(e),p.delete(e)}},h=function(e){var t=e.src?'src="'+e.src+'" ':'src="" ',a=e.sizes?'sizes="'+e.sizes+'" ':"",i=e.srcSetWebp?"<source type='image/webp' srcset=\""+e.srcSetWebp+'" '+a+"/>":"",r=e.srcSet?'srcset="'+e.srcSet+'" ':"",n=e.title?'title="'+e.title+'" ':"",o=e.alt?'alt="'+e.alt+'" ':'alt="" ',s=e.width?'width="'+e.width+'" ':"",d=e.height?'height="'+e.height+'" ':"",l=e.crossOrigin?'crossorigin="'+e.crossOrigin+'" ':"";return"<picture>"+i+"<img "+(e.loading?'loading="'+e.loading+'" ':"")+s+d+a+r+t+o+n+l+'style="position:absolute;top:0;left:0;opacity:1;width:100%;height:100%;object-fit:cover;object-position:center"/></picture>'},m=l.default.forwardRef(function(e,t){var a=e.sizes,i=e.srcSet,r=e.src,n=e.style,o=e.onLoad,c=e.onError,u=e.nativeLazyLoadSupported,A=e.loading,f=(0,s.default)(e,["sizes","srcSet","src","style","onLoad","onError","nativeLazyLoadSupported","loading"]),p={};return u&&(p.loading=A),l.default.createElement("img",(0,d.default)({sizes:a,srcSet:i,src:r},f,{onLoad:o,onError:c,ref:t},p,{style:(0,d.default)({position:"absolute",top:0,left:0,width:"100%",height:"100%",objectFit:"cover",objectPosition:"center"},n)}))});m.propTypes={style:c.default.object,onError:c.default.func,onLoad:c.default.func};var y=function(e){function t(t){var a;a=e.call(this,t)||this;var i=!0,r=!1,n=t.fadeIn,s=!1,d=f(t);!d&&"undefined"!=typeof window&&window.IntersectionObserver&&(i=!1,r=!0),"undefined"!=typeof HTMLImageElement&&"loading"in HTMLImageElement.prototype&&(i=!0,s=!0),"undefined"==typeof window&&(i=!1),t.critical&&(i=!0,r=!1);var c=!(t.critical&&!t.fadeIn);return a.state={isVisible:i,imgLoaded:!1,imgCached:!1,IOSupported:r,fadeIn:n,hasNoScript:c,seenBefore:d,nativeLazyLoadSupported:s},a.imageRef=l.default.createRef(),a.handleImageLoaded=a.handleImageLoaded.bind((0,o.default)((0,o.default)(a))),a.handleRef=a.handleRef.bind((0,o.default)((0,o.default)(a))),a}(0,n.default)(t,e);var a=t.prototype;return a.componentDidMount=function(){if(this.state.isVisible&&"function"==typeof this.props.onStartLoad&&this.props.onStartLoad({wasCached:f(this.props)}),this.props.critical){var e=this.imageRef.current;e&&e.complete&&this.handleImageLoaded()}},a.componentWillUnmount=function(){this.cleanUpListeners&&this.cleanUpListeners()},a.handleRef=function(e){var t=this;this.state.nativeLazyLoadSupported||this.state.IOSupported&&e&&(this.cleanUpListeners=g(e,function(){var e=f(t.props);t.state.isVisible||"function"!=typeof t.props.onStartLoad||t.props.onStartLoad({wasCached:e}),t.setState({isVisible:!0},function(){return t.setState({imgLoaded:e,imgCached:!!t.imageRef.current.currentSrc})})}))},a.handleImageLoaded=function(){var e,t,a;e=this.props,t=u(e),a=t.fluid?t.fluid.src:t.fixed.src,A[a]=!0,this.setState({imgLoaded:!0}),this.state.seenBefore&&this.setState({fadeIn:!1}),this.props.onLoad&&this.props.onLoad()},a.render=function(){var e=u(this.props),t=e.title,a=e.alt,i=e.className,r=e.style,n=void 0===r?{}:r,o=e.imgStyle,s=void 0===o?{}:o,c=e.placeholderStyle,A=void 0===c?{}:c,f=e.placeholderClassName,p=e.fluid,g=e.fixed,y=e.backgroundColor,b=e.durationFadeIn,E=e.Tag,S=e.itemProp,v=(e.critical,u(this.props).loading);var w=this.state.nativeLazyLoadSupported,L=this.state.imgLoaded||!1===this.state.fadeIn,R=!0===this.state.fadeIn&&!this.state.imgCached,j=(0,d.default)({opacity:L?1:0,transition:R?"opacity "+b+"ms":"none"},s),I="boolean"==typeof y?"lightgray":y,B={transitionDelay:b+"ms"},x=(0,d.default)({opacity:this.state.imgLoaded?0:1},R&&B,s,A),C={title:t,alt:this.state.isVisible?"":a,style:x,className:f};if(p){var O=p;return l.default.createElement(E,{className:(i||"")+" gatsby-image-wrapper",style:(0,d.default)({position:"relative",overflow:"hidden"},n),ref:this.handleRef,key:"fluid-"+JSON.stringify(O.srcSet)},l.default.createElement(E,{style:{width:"100%",paddingBottom:100/O.aspectRatio+"%"}}),I&&l.default.createElement(E,{title:t,style:(0,d.default)({backgroundColor:I,position:"absolute",top:0,bottom:0,opacity:this.state.imgLoaded?0:1,right:0,left:0},R&&B)}),O.base64&&l.default.createElement(m,(0,d.default)({src:O.base64},C)),O.tracedSVG&&l.default.createElement(m,(0,d.default)({src:O.tracedSVG},C)),this.state.isVisible&&l.default.createElement("picture",null,O.srcSetWebp&&l.default.createElement("source",{type:"image/webp",srcSet:O.srcSetWebp,sizes:O.sizes}),l.default.createElement(m,{alt:a,title:t,sizes:O.sizes,src:O.src,crossOrigin:this.props.crossOrigin,srcSet:O.srcSet,style:j,ref:this.imageRef,onLoad:this.handleImageLoaded,onError:this.props.onError,itemProp:S,nativeLazyLoadSupported:w,loading:v})),this.state.hasNoScript&&l.default.createElement("noscript",{dangerouslySetInnerHTML:{__html:h((0,d.default)({alt:a,title:t,loading:v},O))}}))}if(g){var z=g,D=(0,d.default)({position:"relative",overflow:"hidden",display:"inline-block",width:z.width,height:z.height},n);return"inherit"===n.display&&delete D.display,l.default.createElement(E,{className:(i||"")+" gatsby-image-wrapper",style:D,ref:this.handleRef,key:"fixed-"+JSON.stringify(z.srcSet)},I&&l.default.createElement(E,{title:t,style:(0,d.default)({backgroundColor:I,width:z.width,opacity:this.state.imgLoaded?0:1,height:z.height},R&&B)}),z.base64&&l.default.createElement(m,(0,d.default)({src:z.base64},C)),z.tracedSVG&&l.default.createElement(m,(0,d.default)({src:z.tracedSVG},C)),this.state.isVisible&&l.default.createElement("picture",null,z.srcSetWebp&&l.default.createElement("source",{type:"image/webp",srcSet:z.srcSetWebp,sizes:z.sizes}),l.default.createElement(m,{alt:a,title:t,width:z.width,height:z.height,sizes:z.sizes,src:z.src,crossOrigin:this.props.crossOrigin,srcSet:z.srcSet,style:j,ref:this.imageRef,onLoad:this.handleImageLoaded,onError:this.props.onError,itemProp:S,nativeLazyLoadSupported:w,loading:v})),this.state.hasNoScript&&l.default.createElement("noscript",{dangerouslySetInnerHTML:{__html:h((0,d.default)({alt:a,title:t,loading:v},z))}}))}return null},t}(l.default.Component);y.defaultProps={fadeIn:!0,durationFadeIn:500,alt:"",Tag:"div",loading:"lazy"};var b=c.default.shape({width:c.default.number.isRequired,height:c.default.number.isRequired,src:c.default.string.isRequired,srcSet:c.default.string.isRequired,base64:c.default.string,tracedSVG:c.default.string,srcWebp:c.default.string,srcSetWebp:c.default.string}),E=c.default.shape({aspectRatio:c.default.number.isRequired,src:c.default.string.isRequired,srcSet:c.default.string.isRequired,sizes:c.default.string.isRequired,base64:c.default.string,tracedSVG:c.default.string,srcWebp:c.default.string,srcSetWebp:c.default.string});y.propTypes={resolutions:b,sizes:E,fixed:b,fluid:E,fadeIn:c.default.bool,durationFadeIn:c.default.number,title:c.default.string,alt:c.default.string,className:c.default.oneOfType([c.default.string,c.default.object]),critical:c.default.bool,crossOrigin:c.default.oneOfType([c.default.string,c.default.bool]),style:c.default.object,imgStyle:c.default.object,placeholderStyle:c.default.object,placeholderClassName:c.default.string,backgroundColor:c.default.oneOfType([c.default.string,c.default.bool]),onLoad:c.default.func,onError:c.default.func,onStartLoad:c.default.func,Tag:c.default.string,itemProp:c.default.string,loading:c.default.oneOf(["auto","lazy","eager"])};var S=y;t.default=S}}]);
//# sourceMappingURL=2-56b3bbd1da6fa5577b1c.js.map