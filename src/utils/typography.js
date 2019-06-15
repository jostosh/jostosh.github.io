import Typography from "typography"
import lincoln from "typography-theme-lincoln"

lincoln.overrideThemeStyles = () => {
  return {
    "a.gatsby-resp-image-link": {
      boxShadow: `none`, 
      textDecoration: `none`,
      border: `none`,
      showCaptions: true,
    },
    'h1,h2': {
      color: '#ff8c00'
    },
    'h3,h4,h5,h6': {
      color: '#cc7000'
    },
    'a': {
      textDecoration: `none`,
      textShadow: 'none',
      backgroundImage: "none",
      color: "#C35C00"
    },
    'em': {
      color: '#cc7000',
      fontStyle: 'normal'
    }
  }
}

// delete Wordpress2016.googleFonts

const typography = new Typography(lincoln)

// Hot reload typography in development.
if (process.env.NODE_ENV !== `production`) {
  typography.injectStyles()
}

export default typography
export const rhythm = typography.rhythm
export const scale = typography.scale
